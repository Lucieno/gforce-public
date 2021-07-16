import argparse
from math import log2, floor
from time import sleep
from itertools import product

import torch
import numpy as np

from config import Config
from fhe import FheBuilder, FheTensor, FheEncTensor, sub_handle
from torch_utils import compare_expected_actual, get_prod, gen_unirand_int_grain, pmod, warming_up_cuda, shuffle_torch, \
    torch_to_list
from timer_utils import NamedTimerInstance


class NetworkSimulator(object):
    bandwidth = None # bps
    basis_latency = 0 # In the unit of second
    total_bit = 0
    total_time = 0

    def __init__(self, bandwidth=10*(10**9), basis_latency=0.001):
        self.bandwidth =  bandwidth
        self.basis_latency = basis_latency

    def simulate(self, data, verbose=False):
        if isinstance(data, FheTensor):
            size_bit = data.padded_num_slot
        elif isinstance(data, np.ndarray):
            size_bit = data.dtype.itemsize * get_prod(data.shape) * 8
        elif isinstance(data, torch.Tensor):
            size_bit = data.storage().element_size() * get_prod(data.shape) * 8
        else:
            raise Exception(f"Unknown Type {type(data)}")

        transfer_time = size_bit / self.bandwidth + self.basis_latency

        self.total_bit += size_bit
        self.total_time += transfer_time

        if verbose:
            print("size_bit:", size_bit)
            print(f"transfer_time: {transfer_time * 10 ** 3} ms")

        sleep(transfer_time)

    def reset(self):
        self.total_bit = 0
        self.total_time = 0


google_vm_simulator = NetworkSimulator(bandwidth=10*(10**9), basis_latency=.001)

class DgkBase(object):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit, name=None):
        self.num_elem = num_elem
        self.q_23 = q_23
        self.q_16 = q_16
        self.work_bit = work_bit
        self.data_bit = data_bit
        self.work_range = 2 ** work_bit
        self.data_range = 2 ** data_bit
        self.decomp_bit_shape = [work_bit, num_elem]
        self.num_work_batch = self.work_bit + 1
        self.sum_shape = [self.num_work_batch, self.num_elem]
        self.nullify_threshold = (self.q_23 - 1) / 2
        self.name = name

    def sub_name(self, sub_name):
        return self.name + '_DgkBitSingleThread_' + sub_name


class SwapOffline(object):
    def __init__(self, num_elem, modulus):
        self.num_elem = num_elem
        self.modulus = modulus

    def generate_random(self):
        return gen_unirand_int_grain(0, self.modulus - 1, self.num_elem)

    def mod_to_modulus(self, input):
        return pmod(input + 2 * self.modulus, self.modulus)

    def check_correctness(self, input):
        print("Check correctness for SwapToServerOffline")
        input = self.mod_to_modulus(input)
        output = self.mod_to_modulus(self.output_s + self.output_c)
        compare_expected_actual(input, output, verbose=True, get_relative=True)


class SwapToServerOffline(SwapOffline):
    def __init__(self, num_elem, modulus):
        super().__init__(num_elem, modulus)

    def offline(self, input_c, output_s=None):
        # client
        self.input_c = input_c.cpu()
        self.r_c = self.generate_random().cpu()

        # server
        if output_s is None:
            self.output_s = self.generate_random().cpu()
        else:
            self.output_s = output_s.cpu()

        self.mask_s = self.mod_to_modulus(self.input_c + self.r_c - self.output_s)

    def online(self, input_s):
        with NamedTimerInstance("SwapToServerOffline Online"):
            # server
            self.combined = self.mod_to_modulus(self.mask_s + input_s.cpu())

            google_vm_simulator.simulate(self.combined.cpu().cuda())

            # client
            self.output_c = self.mod_to_modulus(self.combined - self.r_c)


def swap_to_server_offline_simulation():
    print()
    print("Single Thread Simulation for Swap to Server Offline")
    modulus = 786433
    num_elem = 2 ** 15

    print(f"Number of element: {num_elem}")

    swap_prot = SwapToServerOffline(num_elem, modulus)

    x_c = gen_unirand_int_grain(0, modulus - 1, num_elem).cuda()
    y_s = gen_unirand_int_grain(0, modulus - 1, num_elem).cuda()
    swap_prot.offline(x_c, output_s=y_s)

    x = gen_unirand_int_grain(-modulus // 2 + 1, modulus // 2, num_elem).cuda()
    x_s = pmod(x - x_c, modulus)

    swap_prot.online(x_s)
    swap_prot.check_correctness(x)

    print("totally transferred MB:", google_vm_simulator.total_bit / (2 ** 20 * 8))
    print(f"totally transfer time: {google_vm_simulator.total_time * 10**3} ms")
    google_vm_simulator.reset()


class SwapToClientOffline(SwapOffline):
    def __init__(self, num_elem, modulus):
        super().__init__(num_elem, modulus)

    def offline(self, input_s=None, output_c=None):
        # server
        self.r_s = self.generate_random().cpu()

        # client
        if output_c is None:
            self.output_c = self.generate_random().cpu()
        else:
            self.output_c = output_c.cpu()

        if input_s is None:
            self.mask_c = None
        else:
            self.mask_c = self.mod_to_modulus(input_s.cpu() + self.r_s - self.output_c)

    def online(self, input_c, input_s=None):
        with NamedTimerInstance("SwapToServerOffline Online"):
            # server
            if self.mask_c is None:
                if input_s is None:
                    raise Exception("input_s has to be input online/offline")
                self.mask_c = self.mod_to_modulus(input_s + self.r_s - self.output_c)
                google_vm_simulator.simulate(self.mask_c.cpu())

            # client
            self.combined = self.mod_to_modulus(self.mask_c + input_c.cpu())

            google_vm_simulator.simulate(self.combined.cpu())

            # server
            self.output_s = self.mod_to_modulus(self.combined - self.r_s)


def swap_to_client_offline_simulation():
    print()
    print("Single Thread Simulation for Swap to Server Offline")
    modulus = 786433
    num_elem = 2 ** 15

    print(f"Number of element: {num_elem}")

    swap_prot = SwapToClientOffline(num_elem, modulus)

    x_s = gen_unirand_int_grain(0, modulus - 1, num_elem).cpu()
    y_c = gen_unirand_int_grain(0, modulus - 1, num_elem).cpu()
    swap_prot.offline(x_s, output_c=y_c)

    x = gen_unirand_int_grain(-modulus // 2 + 1, modulus // 2, num_elem).cpu()
    x_c = pmod(x - x_s, modulus)

    swap_prot.online(x_c)
    swap_prot.check_correctness(x)

    print("totally transferred MB:", google_vm_simulator.total_bit / (2 ** 20 * 8))
    print(f"totally transfer time: {google_vm_simulator.total_time * 10**3} ms")
    google_vm_simulator.reset()


class SharesMultSingleThread(object):
    def __init__(self, num_elem, modulus):
        self.num_elem = num_elem
        self.modulus = modulus
        self.fhe_builder = FheBuilder(modulus, Config.n_23)
        self.fhe_builder.generate_keys()

    def generate_random(self):
        return gen_unirand_int_grain(0, self.modulus - 1, self.num_elem)

    def mod_to_modulus(self, input):
        return pmod(input, self.modulus)

    def offline(self):
        # client
        self.u_c = self.generate_random().double().cuda()
        self.v_c = self.generate_random().double().cuda()
        self.fhe_u = self.fhe_builder.build_enc_from_torch(self.u_c)
        self.fhe_v = self.fhe_builder.build_enc_from_torch(self.v_c)

        # server
        self.u_s = self.generate_random().double().cuda()
        self.v_s = self.generate_random().double().cuda()
        self.z_s = self.generate_random().double().cuda()
        self.fhe_u += self.fhe_builder.build_plain_from_torch(self.u_s)
        self.fhe_v += self.fhe_builder.build_plain_from_torch(self.v_s)

        self.fhe_z = self.fhe_u
        self.fhe_z *= self.fhe_v
        self.fhe_z_c = self.fhe_z
        self.fhe_z_c -= self.fhe_builder.build_plain_from_torch(self.z_s)

        # client
        self.z_c = self.fhe_builder.decrypt_to_torch(self.fhe_z_c)

    def online(self, a_s, a_c, b_s, b_c):
        # serve
        self.e_s = self.mod_to_modulus(a_s.double().cuda() - self.u_s).double().cuda()
        self.f_s = self.mod_to_modulus(b_s.double().cuda() - self.v_s).double().cuda()

        google_vm_simulator.simulate(self.e_s.float().cpu().cuda())
        google_vm_simulator.simulate(self.f_s.float().cpu().cuda())

        # client
        self.e_c = self.mod_to_modulus(a_c - self.u_c).double().cuda()
        self.f_c = self.mod_to_modulus(b_c - self.v_c).double().cuda()

        google_vm_simulator.simulate(self.e_c.float().cpu().cuda())
        google_vm_simulator.simulate(self.f_c.float().cpu().cuda())

        # both
        e = self.mod_to_modulus(self.e_s + self.e_c).double().cuda()
        f = self.mod_to_modulus(self.f_s + self.f_c).double().cuda()

        self.c_s = a_s * f + e * b_s + self.z_s - e * f
        self.c_c = a_c * f + e * b_c + self.z_c

    def check_correctness(self, a, b):
        print("Check correctness of <u, v, z> in SharesMult offline preparation")
        expected = self.mod_to_modulus((self.u_s + self.u_c) * (self.v_s + self.v_c))
        actual = self.mod_to_modulus(self.z_s + self.z_c)
        compare_expected_actual(expected, actual, verbose=True, get_relative=True)

        print("Check correctness of SharesMult")
        expected = self.mod_to_modulus(a.double() * b.double())
        actual = self.mod_to_modulus(self.c_s + self.c_c)
        compare_expected_actual(expected, actual, verbose=True, get_relative=True)


def shares_mult_simulation():
    print()
    print("Single Thread Simulation for Shares Multiplication")
    modulus = 786433
    num_elem = 2 ** 17

    print(f"Number of element: {num_elem}")

    prot = SharesMultSingleThread(num_elem, modulus)

    with NamedTimerInstance("Shares Multiplication Offline"):
        prot.offline()

    a = gen_unirand_int_grain(0, modulus - 1, num_elem).cuda()
    a_s = gen_unirand_int_grain(0, modulus - 1, num_elem).cuda()
    a_c = pmod(a - a_s, modulus)
    b = gen_unirand_int_grain(0, modulus - 1, num_elem).cuda()
    b_s = gen_unirand_int_grain(0, modulus - 1, num_elem).cuda()
    b_c = pmod(b - b_s, modulus)

    with NamedTimerInstance("Shares Multiplication Online"):
        prot.online(a_s, a_c, b_s, b_c)
    prot.check_correctness(a, b)

    print("totally transferred MB:", google_vm_simulator.total_bit / (2 ** 20 * 8))
    print(f"totally transfer time: {google_vm_simulator.total_time * 10**3} ms")
    google_vm_simulator.reset()


# server: [x] -> [x + r_s]
# client: [x + r_s] -> [x + r_s]
# server: [x + r_s] -> [x]
class EncRefresherSingleThread(object):
    r_s = None
    def __init__(self, shape, fhe_builder: FheBuilder):
        self.shape = shape
        self.fhe_builder = fhe_builder
        self.modulus = fhe_builder.modulus

    def request(self, enc):
        # server
        self.r_s = gen_unirand_int_grain(0, self.modulus - 1, self.shape)
        def sub_request(sub_enc, sub_r_s):
            sub_enc += self.fhe_builder.build_plain_from_torch(sub_r_s)
            return sub_enc
        fhe_masked = sub_handle(sub_request, enc, self.r_s)

        # client
        def sub_reencrypt(sub_enc):
            sub_dec = self.fhe_builder.decrypt_to_torch(sub_enc)
            sub_refreshed = self.fhe_builder.build_enc_from_torch(sub_dec)
            return sub_refreshed
        with NamedTimerInstance("Client refreshing"):
            refreshed = sub_handle(sub_reencrypt, fhe_masked)

        # server
        def sub_unmask(sub_enc, sub_r_s):
            sub_enc -= self.fhe_builder.build_plain_from_torch(sub_r_s)
            return sub_enc
        unmasked_refreshed = sub_handle(sub_unmask, refreshed, self.r_s)

        return unmasked_refreshed


def test_fhe_refresh():
    print()
    print("Test: Single Thread of FHE refresh: Start")
    modulus, degree = 12289, 2048
    num_batch = 12
    num_elem = 2 ** 15
    fhe_builder = FheBuilder(modulus, degree)
    fhe_builder.generate_keys()

    shape = [num_batch, num_elem]
    tensor = gen_unirand_int_grain(0, modulus - 1, shape)
    refresher = EncRefresherSingleThread(shape, fhe_builder)
    enc = [fhe_builder.build_enc_from_torch(tensor[i]) for i in range(num_batch)]
    refreshed = refresher.request(enc)
    tensor_refreshed = fhe_builder.decrypt_to_torch(refreshed)
    compare_expected_actual(tensor, tensor_refreshed, get_relative=True, name="batch refresh")

    shape = num_elem
    tensor = gen_unirand_int_grain(0, modulus - 1, shape)
    refresher = EncRefresherSingleThread(shape, fhe_builder)
    enc = fhe_builder.build_enc_from_torch(tensor)
    refreshed = refresher.request(enc)
    tensor_refreshed = fhe_builder.decrypt_to_torch(refreshed)
    compare_expected_actual(tensor, tensor_refreshed, get_relative=True, name="batch refresh")

    print()
    print("Test: Single Thread of FHE refresh: End")


class DgkSingleThread(DgkBase):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit, is_shuffle=False, is_trunc=False, divisor_pow=None):
        super(DgkSingleThread, self).__init__(num_elem, q_23, q_16, work_bit, data_bit)
        self.fhe_builder_16 = FheBuilder(q_16, 2048)
        self.fhe_builder_16.generate_keys()
        self.fhe_builder_23 = FheBuilder(q_23, Config.n_23)
        self.fhe_builder_23.generate_keys()
        self.is_shuffle = is_shuffle
        self.is_trunc = is_trunc
        self.divisor_pow = divisor_pow
        if self.is_trunc:
            assert(isinstance(self.divisor_pow, int))
            assert(1 <= self.divisor_pow <= int(floor(log2(q_23))))
            self.mult_after = self.q_23 // (2 ** self.divisor_pow)

    def decomp_to_bit(self, x):
        tmp_x = torch.clone(x).cuda()
        res = torch.zeros([self.work_bit, self.num_elem])
        for i in range(self.work_bit):
            res[i] = pmod(tmp_x, 2)
            tmp_x //= 2
        return res

    def xor_fhe(self, alpha_i, fhe_enc, mask_s, modulus, change_sign, multi_after = 1):
        if modulus == self.q_16:
            fhe_builder = self.fhe_builder_16
        elif modulus == self.q_23:
            fhe_builder = self.fhe_builder_23
        else:
            raise Exception(f"Unknown modulus: {modulus}")
        zeros = torch.zeros_like(alpha_i)
        mult = torch.where(alpha_i == change_sign, modulus - multi_after + zeros, multi_after + zeros)
        bias = torch.where(alpha_i == change_sign, 1 + zeros, zeros)
        fhe_mult = fhe_builder.build_plain_from_torch(mult)
        fhe_bias = fhe_builder.build_plain_from_torch(bias)
        fhe_mask_s = fhe_builder.build_plain_from_torch(mask_s)
        fhe_enc *= fhe_mult
        fhe_enc += fhe_bias
        fhe_enc += fhe_mask_s
        del fhe_mult, fhe_bias, fhe_mask_s
        del mult, bias, zeros
        return fhe_enc

    def xor_alpha_known_offline_plain(self, alpha_i, beta_i_c, mask_s):
        res = torch.where(alpha_i == 1, 1 - beta_i_c, beta_i_c)
        res += mask_s
        res.fmod_(self.q_16)
        return res

    def xor_alpha_known_offline(self, alpha_i, fhe_beta_i_c, mask_s):
        assert(len(alpha_i) == self.work_bit)
        assert(len(fhe_beta_i_c) == self.work_bit)
        assert(len(mask_s) == self.work_bit)
        res = [self.xor_fhe(alpha_i[i], fhe_beta_i_c[i], mask_s[i], self.q_16, 1) for i in range(self.work_bit)]
        # for i in range(self.work_bit):
        #     res[i] -= self.fhe_d_c[i]
        return res

    def xor_alpha_known_online(self, alpha_i, beta_i_s, mask_s, modulus):
        res = torch.where(alpha_i == 1, -beta_i_s, beta_i_s)
        res += modulus - mask_s
        # res -= self.d_s
        res.fmod_(modulus)
        return res

    def xor_delta_known_offline_plain(self, alpha_i, beta_i_c, mask_s):
        res = torch.where(alpha_i == 1, beta_i_c, 1 - beta_i_c)
        res += mask_s
        res.fmod_(self.q_23)
        return res

    def xor_delta_known_offline(self, alpha_i, fhe_beta_i_c, mask_s):
        return self.xor_fhe(alpha_i, fhe_beta_i_c, mask_s, self.q_23, 0)

    def xor_delta_known_online(self, alpha_i, beta_i_s, mask_s, modulus):
        res = torch.where(alpha_i == 1, beta_i_s, -beta_i_s)
        res += modulus - mask_s
        res.fmod_(modulus)
        return res

    def xor_trunc_offline(self, delta_s, fhe_delta_c, mask_s):
        return self.xor_fhe(delta_s, fhe_delta_c, mask_s, self.q_23, 0, multi_after=self.mult_after)

    def xor_trunc_online(self, delta_s, beta_c, mask_s):
        modulus = self.q_23
        res = torch.where(delta_s == 0, -beta_c, beta_c) * self.mult_after
        res = pmod(res - mask_s, modulus)
        return res

    def sum_c_i_common(self, alpha_beta_xor_share):
        sum_xor = torch.zeros(self.sum_shape).cuda()
        # the last row of sum_xor is c_{-1}, which helps check the case with x == y
        for i in range(self.work_bit - 1)[::-1]:
            sum_xor[i] = sum_xor[i + 1] + alpha_beta_xor_share[i + 1]
        return sum_xor

    def sum_c_i_offline_plain(self, delta_a, beta_i_c, alpha_beta_xor_c, s, alpha_i, ci_mask_s, mult_mask_s, shuffle_order):
        sum_xor = self.sum_c_i_common(alpha_beta_xor_c)
        sum_xor[self.work_bit] = sum_xor[0] + alpha_beta_xor_c[0] + delta_a
        for i in range(self.work_bit)[::-1]:
            sum_xor[i] = 3 * sum_xor[i] - beta_i_c[i] + s + alpha_i[i]
        sum_xor = sum_xor.double() * mult_mask_s
        sum_xor += ci_mask_s
        sum_xor = pmod(sum_xor, self.q_16).float().cuda()
        if self.is_shuffle:
            sum_xor = torch.zeros_like(sum_xor).scatter_(0, shuffle_order, sum_xor.cuda())
        return sum_xor

    def generate_fhe_shuffled(self, shuffle_order, enc):
        num_batch = self.num_work_batch
        fhe_builder = self.fhe_builder_16
        res = [fhe_builder.build_enc(self.num_elem) for i in range(num_batch)]
        zeros = torch.zeros(self.num_elem).cuda()
        for dst, src in product(range(num_batch), range(num_batch)):
            mask = torch.where(shuffle_order[src, :] == dst, 1 + zeros, zeros)
            fhe_mask = fhe_builder.build_plain_from_torch(mask)
            enc_tmp = enc[src].copy()
            enc_tmp *= fhe_mask
            res[dst] += enc_tmp
            del enc_tmp, fhe_mask, mask
        return res

    def sum_c_i_offline(self, delta_a, fhe_beta_i_c, fhe_alpha_beta_xor_c, s, alpha_i, ci_mask_s, mult_mask_s, shuffle_order):
        # the last row of sum_xor is c_{-1}, which helps check the case with x == y
        fhe_builder = self.fhe_builder_16
        fhe_sum_xor = [fhe_builder.build_enc(self.num_elem) for i in range(self.num_work_batch)]
        for i in range(self.work_bit - 1)[::-1]:
            fhe_sum_xor[i] += fhe_sum_xor[i + 1]
            fhe_sum_xor[i] += fhe_alpha_beta_xor_c[i + 1]
        fhe_sum_xor[self.work_bit] += fhe_sum_xor[0]
        fhe_sum_xor[self.work_bit] += fhe_alpha_beta_xor_c[0]
        fhe_sum_xor[self.work_bit] += fhe_builder.build_plain_from_torch(delta_a)

        for i in range(self.work_bit)[::-1]:
            fhe_mult_3 = fhe_builder.build_plain_from_torch(pmod(3 * mult_mask_s[i].cpu(), self.q_16))
            fhe_mult_mask_s = fhe_builder.build_plain_from_torch(mult_mask_s[i])
            fhe_s = fhe_builder.build_plain_from_torch(s * mult_mask_s[i])
            fhe_alpha_i = fhe_builder.build_plain_from_torch(alpha_i[i] * mult_mask_s[i])
            fhe_ci_mask_s = fhe_builder.build_plain_from_torch(ci_mask_s[i])
            fhe_beta_i_c[i] *= fhe_mult_mask_s
            fhe_sum_xor[i] *= fhe_mult_3
            fhe_sum_xor[i] -= fhe_beta_i_c[i]
            fhe_sum_xor[i] += fhe_s
            fhe_sum_xor[i] += fhe_alpha_i
            fhe_sum_xor[i] += fhe_ci_mask_s
            # fhe_sum_xor[i] += self.fhe_d_c[i]

        fhe_mult_mask_s = fhe_builder.build_plain_from_torch(mult_mask_s[self.work_bit])
        fhe_ci_mask_s = fhe_builder.build_plain_from_torch(ci_mask_s[self.work_bit])
        fhe_sum_xor[self.work_bit] *= fhe_mult_mask_s
        fhe_sum_xor[self.work_bit] += fhe_ci_mask_s

        if self.is_shuffle:
            refresher = EncRefresherSingleThread(self.sum_shape, fhe_builder)
            fhe_sum_xor = refresher.request(fhe_sum_xor)
            fhe_sum_xor = self.generate_fhe_shuffled(shuffle_order, fhe_sum_xor)

        return fhe_sum_xor

    def shuffle_torch(self, torch_tensor, shuffling_order):
        return torch.zeros_like(torch_tensor).scatter_(0, shuffling_order, torch_tensor.cuda())

    def sum_c_i_online(self, beta_i_s, alpha_beta_xor_s, ci_mask_s, mult_mask_s, shuffle_order):
        sum_xor = self.sum_c_i_common(alpha_beta_xor_s)
        sum_xor[self.work_bit] = sum_xor[0] + alpha_beta_xor_s[0]
        for i in range(self.work_bit)[::-1]:
            # sum_xor[i] = 3 * sum_xor[i] - beta_i_s[i] + self.d_s[i]
            sum_xor[i] = 3 * sum_xor[i] - beta_i_s[i]
        sum_xor = sum_xor.double() * mult_mask_s
        sum_xor -= ci_mask_s
        sum_xor = pmod(sum_xor, self.q_16).float().cuda()
        if self.is_shuffle:
            sum_xor = shuffle_torch(sum_xor, shuffle_order)
        return sum_xor

    def mod_div_offline(self, r):
        fhe_builder = self.fhe_builder_23

        # client
        self.pre_mod_div_c = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda()
        # self.pre_mod_div_c = torch.ones(self.num_elem).cuda()
        fhe_correct_mod_div_work = fhe_builder.build_enc_from_torch(self.pre_mod_div_c)

        # server
        self.elem_zeros = torch.zeros(self.num_elem).cuda()
        self.correct_mod_div_work_mult = torch.where((r < self.nullify_threshold),
                                                     self.elem_zeros,
                                                     self.elem_zeros + self.q_23 // self.work_range).double()
        self.correct_mod_div_work_mask_s = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda()
        # self.correct_mod_div_work_mask_s = torch.zeros(self.num_elem).cuda()
        fhe_mult = fhe_builder.build_plain_from_torch(self.correct_mod_div_work_mult)
        fhe_bias = fhe_builder.build_plain_from_torch(self.correct_mod_div_work_mask_s)
        fhe_correct_mod_div_work *= fhe_mult
        fhe_correct_mod_div_work += fhe_bias

        # client
        fhe_builder.noise_budget(fhe_correct_mod_div_work, name="fhe_correct_mod_div_work")
        self.correct_mod_div_work_c = fhe_builder.decrypt_to_torch(fhe_correct_mod_div_work)

    def mod_div_online(self, z):
        # client
        pre_correct_mod_div_s = torch.where(z < self.nullify_threshold, self.elem_zeros + 1, self.elem_zeros)
        pre_correct_mod_div_s = pmod(pre_correct_mod_div_s - self.pre_mod_div_c, self.q_23)
        google_vm_simulator.simulate(pre_correct_mod_div_s.type(torch.float).cpu().cuda())

        # server
        self.correct_mod_div_work_s = pmod(
            self.correct_mod_div_work_mult * pre_correct_mod_div_s - self.correct_mod_div_work_mask_s, self.q_23)

        expected = torch.where(self.r > self.z, self.q_23//self.work_range + self.elem_zeros, self.elem_zeros)
        actual = pmod(self.correct_mod_div_work_s + self.correct_mod_div_work_c, self.q_23)
        compare_expected_actual(expected, actual, get_relative=True, name="mod_div_online")

    def d_offline(self, r, alpha_i):
        return
        fhe_builder = self.fhe_builder_16
        num_batch = self.decomp_bit_shape[0]

        # client
        self.pre_d_c = gen_unirand_int_grain(0, self.q_16 - 1, self.decomp_bit_shape).cuda()
        # self.pre_d_c = torch.zeros(self.decomp_bit_shape).cuda()
        fhe_d_c = [fhe_builder.build_enc_from_torch(self.pre_d_c[i]) for i in range(num_batch)]

        alpha_hat = pmod(r - self.q_23, self.work_range)
        self.alpha_i_hat = self.decomp_to_bit(alpha_hat).cuda()
        self.decomp_bit_zeros = torch.zeros(self.decomp_bit_shape).cuda()
        self.nullify_mult = torch.where((r < self.nullify_threshold) | (self.alpha_i_hat == alpha_i),
                                        self.decomp_bit_zeros, self.decomp_bit_zeros + 1)
        self.d_mask_s = gen_unirand_int_grain(0, self.q_16 - 1, self.decomp_bit_shape).cuda()
        # self.d_mask_s = torch.zeros(self.decomp_bit_shape).cuda()
        fhe_mult = [fhe_builder.build_plain_from_torch(self.nullify_mult[i]) for i in range(num_batch)]
        fhe_bias = [fhe_builder.build_plain_from_torch(self.d_mask_s[i]) for i in range(num_batch)]
        for i in range(num_batch):
            fhe_d_c[i] *= fhe_mult[i]
            fhe_d_c[i] += fhe_bias[i]
        self.fhe_d_c = fhe_d_c

    def d_online(self, z):
        return
        # client
        pre_s = torch.where(z < self.nullify_threshold, self.decomp_bit_zeros + 1, self.decomp_bit_zeros)
        pre_s = pmod(pre_s - self.pre_d_c, self.q_16) # send

        google_vm_simulator.simulate(pre_s.type(torch.float).cpu().cuda())

        # server
        self.d_s = pmod(self.nullify_mult * pre_s - self.d_mask_s, self.q_16)

        expected = ((self.r > z) & (self.alpha_i != self.alpha_i_hat)) * 1
        d_c = self.fhe_builder_16.decrypt_to_torch(self.fhe_d_c)
        actual = pmod(self.d_s + d_c, self.q_16)
        compare_expected_actual(expected, actual, get_relative=True, name="d_online")

    def offline(self, y_sub_x_s=None, is_check_internal=False):
        ## server
        self.delta_a = gen_unirand_int_grain(0, 1, self.num_elem).cuda()
        self.s = 1 - 2 * self.delta_a

        # self.r = gen_unirand_int_grain(0, 2 ** (self.work_bit) - 1, self.num_elem).cuda()
        self.r = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda()
        self.nullify_mult = torch.where(self.r > (self.q_23 - 1) // 2, 0 + torch.zeros_like(self.r), 1 + torch.zeros_like(self.r))
        self.alpha = pmod(self.r, self.work_range)
        self.alpha_i = self.decomp_to_bit(self.alpha).cuda()
        self.r_work_s = (self.r // self.work_range)

        self.y_sub_x_s = y_sub_x_s
        if self.y_sub_x_s is not None:
            self.z_s = pmod(self.y_sub_x_s + self.work_range + self.r, self.q_23)

        ## client
        self.beta_i_c = gen_unirand_int_grain(0, self.q_16 - 1, self.decomp_bit_shape).cuda()
        self.delta_b_c = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda()
        self.z_work_c = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda()

        fhe_beta_i_c = [self.fhe_builder_16.build_enc_from_torch(self.beta_i_c[i]) for i in range(len(self.beta_i_c))] # send
        fhe_delta_b_c = self.fhe_builder_23.build_enc_from_torch(self.delta_b_c) # send
        if not self.is_trunc:
            fhe_z_work_c = self.fhe_builder_23.build_enc_from_torch(self.z_work_c) # send

        ## server
        self.beta_i_mask_s = gen_unirand_int_grain(0, self.q_16 - 1, self.decomp_bit_shape).cuda()
        # self.beta_i_mask_s = torch.zeros_like(self.beta_i_mask_s).cuda()
        self.ci_mask_s = gen_unirand_int_grain(0, self.q_16 - 1, [self.work_bit + 1, self.num_elem]).cuda()
        self.ci_mult_mask_s = gen_unirand_int_grain(1, self.q_16 - 1, [self.work_bit + 1, self.num_elem]).cuda()
        self.shuffle_order = torch.rand([self.work_bit + 1, self.num_elem]).argsort(dim=0).cuda()

        self.mod_div_offline(self.r)
        self.d_offline(self.r, self.alpha_i)
        fhe_beta_i_c_for_sum_c = [fhe_beta_i_c[i].copy() for i in range(len(fhe_beta_i_c))]
        fhe_alpha_beta_xor_c = self.xor_alpha_known_offline(self.alpha_i, fhe_beta_i_c, self.beta_i_mask_s)

        if is_check_internal:
            alpha_beta_xor_c_expected = pmod(self.xor_alpha_known_offline_plain(self.alpha_i, self.beta_i_c, self.beta_i_mask_s), self.q_16)
            alpha_beta_xor_c_actual = self.fhe_builder_16.decrypt_to_torch(fhe_alpha_beta_xor_c)
            compare_expected_actual(alpha_beta_xor_c_expected, alpha_beta_xor_c_actual, get_relative=True, name="alpha_beta_xor_c")

        refresher_alpha_beta_xor_c = EncRefresherSingleThread(self.decomp_bit_shape, self.fhe_builder_16)
        fhe_alpha_beta_xor_c = refresher_alpha_beta_xor_c.request(fhe_alpha_beta_xor_c)

        fhe_c_i_c = self.sum_c_i_offline(self.delta_a, fhe_beta_i_c_for_sum_c, fhe_alpha_beta_xor_c, self.s,
                                         self.alpha_i, self.ci_mask_s, self.ci_mult_mask_s, self.shuffle_order)

        if is_check_internal:
            c_i_c_expected = self.sum_c_i_offline_plain(self.delta_a, self.beta_i_c, alpha_beta_xor_c_expected, self.s,
                                             self.alpha_i, self.ci_mask_s, self.ci_mult_mask_s, self.shuffle_order)
            c_i_c_actual = self.fhe_builder_16.decrypt_to_torch(fhe_c_i_c)
            compare_expected_actual(pmod(c_i_c_expected, self.q_16), c_i_c_actual, get_relative=True, name="c_i_c")

        self.c_i_c = self.fhe_builder_16.decrypt_to_torch(fhe_c_i_c)

        self.delta_xor_mask_s = gen_unirand_int_grain(0, self.q_16 - 1, self.num_elem).cuda()

        if self.is_trunc:
            fhe_delta_xor_c = self.xor_trunc_offline(self.delta_a, fhe_delta_b_c, self.delta_xor_mask_s)
            self.overcarry_c = self.fhe_builder_23.decrypt_to_torch(fhe_delta_xor_c)
            self.r_div_d = self.r // (2 ** self.divisor_pow)

        else:
            fhe_delta_xor_c = self.xor_delta_known_offline(self.delta_a, fhe_delta_b_c, self.delta_xor_mask_s)

            self.dgk_x_leq_y_mask_s = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda()
            fhe_z_work_c -= fhe_delta_xor_c
            fhe_z_work_c -= self.fhe_builder_23.build_plain_from_torch(self.dgk_x_leq_y_mask_s)
            self.dgk_x_leq_y_c = self.fhe_builder_23.decrypt_to_torch(fhe_z_work_c)
            self.dgk_x_leq_y_c = pmod(self.dgk_x_leq_y_c + self.correct_mod_div_work_c, self.q_23)
            self.delta_xor_c = self.fhe_builder_23.decrypt_to_torch(fhe_delta_xor_c)

    def online(self, y_sub_x_c, y_sub_x_s=None):
        ## client
        if self.y_sub_x_s is not None:
            self.z = pmod(self.z_s + y_sub_x_c + 2 * self.q_23, self.q_23)
        elif y_sub_x_s is not None:
            self.z_s = pmod(y_sub_x_s + self.work_range + self.r, self.q_23)
            google_vm_simulator.simulate(self.z_s.float().cpu().cuda())
            self.z = pmod(self.z_s + y_sub_x_c + 2 * self.q_23, self.q_23)
        else:
            raise Exception("y_sub_x_s has to be input during online/offline")

        self.beta = pmod(self.z, self.work_range)
        self.beta_i = self.decomp_to_bit(self.beta).cuda()
        self.beta_i_s = pmod(self.beta_i.cuda() - self.beta_i_c.cuda(), self.q_16)

        google_vm_simulator.simulate(self.beta_i_s.type(torch.int16).cpu().cuda())

        ## server
        self.mod_div_online(self.z)
        self.d_online(self.z)
        self.alpha_beta_xor_s = self.xor_alpha_known_online(self.alpha_i, self.beta_i_s, self.beta_i_mask_s, self.q_16)
        self.c_i_s = self.sum_c_i_online(self.beta_i_s, self.alpha_beta_xor_s, self.ci_mask_s, self.ci_mult_mask_s, self.shuffle_order)

        google_vm_simulator.simulate(self.c_i_s.type(torch.int16).cpu().cuda())

        ## client
        self.c_i = pmod(self.c_i_s + self.c_i_c, self.q_16)
        self.check_zeros = torch.where(self.c_i == 0, torch.ones_like(self.c_i), torch.zeros_like(self.c_i))
        self.delta_b = torch.where(torch.sum(self.check_zeros, 0) > 0,
                              torch.ones(self.num_elem).cuda(), torch.zeros(self.num_elem).cuda())

        self.delta_b_s = pmod(self.delta_b - self.delta_b_c, self.q_23)
        google_vm_simulator.simulate(self.delta_b_s.float().cpu().cuda())

        if not self.is_trunc:
            self.z_work_s = pmod(self.z // self.work_range - self.z_work_c, self.q_23)
            google_vm_simulator.simulate(self.z_work_s.float().cpu().cuda())

        ## server
        if self.is_trunc:
            self.delta_xor_s = self.xor_trunc_online(self.delta_a, self.delta_b_s, self.delta_xor_mask_s)
            self.trunc_s = pmod(self.delta_xor_s - self.div_r_r, self.q_23)
        else:
            self.delta_xor_s = self.xor_delta_known_online(self.delta_a, self.delta_b_s, self.delta_xor_mask_s, self.q_23)
            # self.dgk_x_leq_y_s = self.z_work_s - ((self.r // self.work_range) + self.delta_xor_s) + self.dgk_x_leq_y_mask_s
            self.dgk_x_leq_y_s = pmod(
                self.z_work_s - (self.r_work_s + self.delta_xor_s) + self.correct_mod_div_work_s + self.dgk_x_leq_y_mask_s,
                self.q_23)

            self.delta_xor_recon = pmod(self.delta_xor_s + self.delta_xor_c, self.q_23)

            self.expected_b_less_a = (self.beta < self.alpha)
            self.dgk_b_less_a = torch.where(self.delta_a == 1, self.delta_b, 1 - self.delta_b)

    def check_correctness(self, x, y):
        x = torch.where(x < self.q_23 // 2, x, x - self.q_23).cuda()
        y = torch.where(y < self.q_23 // 2, y, y - self.q_23).cuda()
        self.expected_x_leq_y = (x <= y)
        self.dgk_x_leq_y = pmod(
            (self.z // self.work_range) - ((self.r // self.work_range) + self.dgk_b_less_a),
            self.q_23)
        self.dgk_x_leq_y = torch.where(self.r > self.z, self.dgk_x_leq_y + self.q_23//self.work_range, self.dgk_x_leq_y)
        # self.dgk_x_leq_y = torch.where(self.r > self.z, self.dgk_x_leq_y + 1, self.dgk_x_leq_y)
        self.dgk_x_leq_y = pmod(self.dgk_x_leq_y, self.q_23)
        self.dgk_x_leq_y_recon = pmod(self.dgk_x_leq_y_s + self.dgk_x_leq_y_c, self.q_23)

        # compare_expected_actual(self.expected_b_less_a, self.dgk_b_less_a, name="b_less_a", get_relative=True)
        compare_expected_actual(self.expected_x_leq_y, self.dgk_x_leq_y, get_relative=True, name="x_leq_y")
        # err_indices = (self.expected_x_leq_y != self.dgk_x_leq_y)
        # print("expected:", self.expected_x_leq_y[err_indices][:10])
        # print("actual:", self.dgk_x_leq_y[err_indices][:10])
        # print("z:", self.z[err_indices][:10])
        # print("r:", self.r[err_indices][:10])
        # print("num of err:", torch.sum(err_indices).item())
        # print("num of over-carry:", torch.sum(self.r > self.z).item())
        # print("num of err but not over-carry:", torch.sum(err_indices & ~(self.r > self.z)).item())
        print("DGK x <= y")
        compare_expected_actual(self.expected_x_leq_y, self.dgk_x_leq_y_recon, verbose=True, get_relative=True)


def dgk_simulation():
    print()
    print("Single Thread Simulation for Dgk")
    data_bit = 19
    work_bit = 19
    data_range = 2 ** data_bit
    # q_23 = 786433
    # q_23 = 1785857
    q_23 = 7340033
    # q_23 = 8273921
    q_16 = 12289
    num_elem = 2 ** 17

    print(f"Number of element: {num_elem}")

    dgk = DgkSingleThread(num_elem, q_23, q_16, work_bit, data_bit)

    x = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem).cuda()
    y = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem).cuda()
    x_s = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem).cuda()
    y_s = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem).cuda()
    y_sub_x_s = pmod(y_s - x_s, q_23)

    with NamedTimerInstance("Dgk Offline"):
        dgk.offline(y_sub_x_s=y_sub_x_s)

    x_c = pmod(x - x_s + q_23, q_23)
    y_c = pmod(y - y_s + q_23, q_23)
    y_sub_x_c = pmod(y_c - x_c, q_23)

    with NamedTimerInstance("Dgk Online"):
        dgk.online(y_sub_x_c, y_sub_x_s)
    dgk.check_correctness(x, y)

    print("totally transferred MB:", google_vm_simulator.total_bit / (2 ** 20 * 8))
    print(f"totally transfer time: {google_vm_simulator.total_time * 10**3} ms")
    google_vm_simulator.reset()


class MaxDgkSingleThread(DgkBase):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit):
        super(MaxDgkSingleThread, self).__init__(num_elem, q_23, q_16, work_bit, data_bit)
        self.dgk = DgkSingleThread(num_elem, q_23, q_16, work_bit, data_bit)
        self.fhe_builder = FheBuilder(q_23, 4096)
        self.fhe_builder.generate_keys()

    def offline(self, x_s=None, y_s=None):
        # server
        self.multipled_mask = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda()
        self.fhe_multipled_mask = self.fhe_builder.build_plain_from_torch(self.multipled_mask)

        # both
        self.y_s = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda() if y_s is None else y_s
        self.x_s = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda() if x_s is None else x_s
        self.dgk.offline(y_sub_x_s=pmod(y_s-x_s, self.q_23))

        # client
        self.fhe_random_sub_c = self.fhe_builder.build_enc_from_torch(self.dgk.dgk_x_leq_y_c)

        # server
        self.random_sub_s = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda()
        self.max_s =  pmod(self.random_sub_s - self.multipled_mask, self.q_23)

        self.fhe_y_s = self.fhe_builder.build_plain_from_torch(self.y_s)
        self.fhe_x_s = self.fhe_builder.build_plain_from_torch(self.x_s)

        self.y_sub_x_s = pmod(self.y_s - self.x_s, self.q_23)

        self.fhe_y_sub_x = self.fhe_builder.build_plain_from_torch(self.y_sub_x_s)
        self.fhe_random_sub_s = self.fhe_builder.build_plain_from_torch(self.random_sub_s)
        self.fhe_random_sub_c *= self.fhe_y_sub_x
        self.fhe_random_sub_c -= self.fhe_random_sub_s

        # client
        self.random_sub_c = self.fhe_builder.decrypt_to_torch(self.fhe_random_sub_c)

    def online(self, x_c, y_c, x=None, y=None):
        with NamedTimerInstance("DgkMax Online"):
            self.x_c = x_c
            self.y_c = y_c
            y_sub_x_c = pmod(y_c - x_c + 2 * self.q_23, self.q_23)

            self.dgk.online(y_sub_x_c)
            self.fhe_enc_x = self.fhe_builder.build_enc_from_torch(x_c)
            self.fhe_enc_y = self.fhe_builder.build_enc_from_torch(y_c)

            google_vm_simulator.simulate(self.fhe_enc_x)
            google_vm_simulator.simulate(self.fhe_enc_y)

            # server
            with NamedTimerInstance("DgkMax Server Online"):
                self.fhe_enc_x += self.fhe_x_s
                self.fhe_enc_y += self.fhe_y_s
                self.fhe_plain_dgk_x_leq_y_s = self.fhe_builder.build_plain_from_torch(self.dgk.dgk_x_leq_y_s)
                self.fhe_enc_y_sub_x = self.fhe_enc_y
                self.fhe_enc_y_sub_x -= self.fhe_enc_x
                self.fhe_multiplied = self.fhe_enc_y_sub_x
                self.fhe_multiplied *= self.fhe_plain_dgk_x_leq_y_s
                self.fhe_multiplied += self.fhe_enc_x
                self.fhe_multiplied += self.fhe_multipled_mask

            google_vm_simulator.simulate(self.fhe_multiplied)

            # client
            with NamedTimerInstance("DgkMax Client Online"):
                self.multipled = self.fhe_builder.decrypt_to_torch(self.fhe_multiplied)
                self.random_unmask = pmod(
                    (self.y_c - self.x_c).double() * self.dgk.dgk_x_leq_y_c.double() + self.random_sub_c,
                    self.q_23)
                self.max_c = pmod(self.multipled + self.random_unmask, self.q_23)

        if x is not None and y is not None:
            self.check_correctness(x, y)

    def check_correctness(self, x, y):
        x = torch.where(x < self.q_23 // 2, x, x - self.q_23).cuda()
        y = torch.where(y < self.q_23 // 2, y, y - self.q_23).cuda()
        max_expected = pmod(torch.max(x, y) + self.q_23, self.q_23)
        max_recon = pmod(self.max_s.double() + self.max_c.double() + 2 * self.q_23, self.q_23)
        print("MaxDgkSingleThread")
        compare_expected_actual(max_expected, max_recon, verbose=True, get_relative=True)


def dgk_max_simulation():
    print()
    print("Single Thread Simulation for Dgk Max")
    data_bit = 17
    work_bit = 17
    data_range = 2 ** data_bit
    q_23 = 786433
    q_16 = 12289
    num_elem = 2 ** 15

    print(f"Number of element: {num_elem}")

    max_dgk = MaxDgkSingleThread(num_elem, q_23, q_16, work_bit, data_bit)

    x_s = gen_unirand_int_grain(0, q_23 - 1, num_elem).cuda()
    y_s = gen_unirand_int_grain(0, q_23 - 1, num_elem).cuda()
    max_dgk.offline(x_s=x_s, y_s=y_s)

    x = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem).cuda()
    y = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem).cuda()
    x_c = pmod(x - max_dgk.x_s + 2 * q_23, q_23)
    y_c = pmod(y - max_dgk.y_s + 2 * q_23, q_23)

    max_dgk.online(x_c, y_c)
    max_dgk.check_correctness(x, y)

    print("totally transferred MB:", google_vm_simulator.total_bit / (2 ** 20 * 8))
    print(f"totally transfer time: {google_vm_simulator.total_time * 10**3} ms")
    google_vm_simulator.reset()


class MaxDgkSharesMultSingleThread(DgkBase):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit):
        super(MaxDgkSharesMultSingleThread, self).__init__(num_elem, q_23, q_16, work_bit, data_bit)
        self.dgk = DgkSingleThread(num_elem, q_23, q_16, work_bit, data_bit)
        self.shares_mult = SharesMultSingleThread(num_elem, q_23)
        self.fhe_builder = FheBuilder(q_23, 4096)
        self.fhe_builder.generate_keys()
        self.is_server_input_offline = False

    def offline(self, x_s=None, y_s=None):
        # server
        self.multipled_mask = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda()
        self.fhe_multipled_mask = self.fhe_builder.build_plain_from_torch(self.multipled_mask)

        # both
        if x_s is not None and y_s is not None:
            self.y_sub_x_s = pmod(y_s - x_s, self.q_23)
            self.is_server_input_offline = True
        else:
            self.y_sub_x_s = None
        self.dgk.offline(y_sub_x_s=self.y_sub_x_s)
        self.shares_mult.offline()

    def online(self, x_c, y_c, x_s=None, y_s=None, x=None, y=None):
        with NamedTimerInstance("DgkMax Online"):
            # server
            if (x_s is not None or y_s is not None) and self.is_server_input_offline:
                raise Exception("Do not input server shares in both offline and online phase.")
            if (x_s is not None or y_s is not None):
                self.x_s = x_s
                self.y_s = y_s
                self.y_sub_x_s = pmod(self.y_s - self.x_s, self.q_23)

            # client
            self.x_c = x_c
            self.y_c = y_c
            self.y_sub_x_c = pmod(y_c - x_c, self.q_23)

            # both
            self.dgk.online(self.y_sub_x_c, y_sub_x_s=None if self.is_server_input_offline else self.y_sub_x_s)
            self.shares_mult.online(self.y_sub_x_s, self.y_sub_x_c, self.dgk.dgk_x_leq_y_s, self.dgk.dgk_x_leq_y_c)

            self.max_s = pmod(self.shares_mult.c_s + self.x_s, self.q_23)
            self.max_c = pmod(self.shares_mult.c_c + self.x_c, self.q_23)

        if x is not None and y is not None:
            self.check_correctness(x, y)

    def check_correctness(self, x, y):
        x = torch.where(x < self.q_23 // 2, x, x - self.q_23).cuda()
        y = torch.where(y < self.q_23 // 2, y, y - self.q_23).cuda()
        max_expected = pmod(torch.max(x, y) + self.q_23, self.q_23)
        max_recon = pmod(self.max_s.double() + self.max_c.double() + 2 * self.q_23, self.q_23)
        print("MaxDgkSingleThread")
        compare_expected_actual(max_expected, max_recon, verbose=True, get_relative=True)


def dgk_max_shares_mults_simulation():
    print()
    print("Single Thread Simulation for Dgk Max")
    data_bit = 17
    work_bit = 17
    data_range = 2 ** data_bit
    q_23 = 786433
    q_16 = 12289
    num_elem = 2 ** 17

    print(f"Number of element: {num_elem}")

    prot = MaxDgkSharesMultSingleThread(num_elem, q_23, q_16, work_bit, data_bit)

    x_s = gen_unirand_int_grain(0, q_23 - 1, num_elem).cuda()
    y_s = gen_unirand_int_grain(0, q_23 - 1, num_elem).cuda()
    prot.offline()

    x = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem).cuda()
    y = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem).cuda()
    x_c = pmod(x - x_s, q_23)
    y_c = pmod(y - y_s, q_23)

    prot.online(x_c, y_c, x_s, y_s)
    prot.check_correctness(x, y)

    print("totally transferred MB:", google_vm_simulator.total_bit / (2 ** 20 * 8))
    print(f"totally transfer time: {google_vm_simulator.total_time * 10**3} ms")
    google_vm_simulator.reset()


class ReluDgkSingleThread(DgkBase):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit):
        super(ReluDgkSingleThread, self).__init__(num_elem, q_23, q_16, work_bit, data_bit)
        self.dgk = DgkSingleThread(num_elem, q_23, q_16, work_bit, data_bit)
        self.fhe_builder = FheBuilder(q_23, 4096)
        self.fhe_builder.generate_keys()

    def offline(self, x_s=None):
        # server
        self.multipled_mask = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda()
        self.fhe_multipled_mask = self.fhe_builder.build_plain_from_torch(self.multipled_mask)

        # both
        self.dgk.offline(y_sub_x_s=x_s)

        # client
        self.fhe_random_sub_c = self.fhe_builder.build_enc_from_torch(self.dgk.dgk_x_leq_y_c)

        # server
        # self.x_s = self.dgk.y_s - self.dgk.x_s
        self.x_s = self.dgk.y_sub_x_s
        self.random_sub_s = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda()
        self.max_s = pmod(self.random_sub_s - self.multipled_mask, self.q_23)

        self.y_sub_x_s = pmod(self.x_s + self.q_23, self.q_23)

        self.fhe_x_s = self.fhe_builder.build_plain_from_torch(self.x_s)
        self.fhe_y_sub_x = self.fhe_builder.build_plain_from_torch(self.y_sub_x_s)
        self.fhe_random_sub_s = self.fhe_builder.build_plain_from_torch(self.random_sub_s)
        self.fhe_random_sub_c *= self.fhe_y_sub_x
        self.fhe_random_sub_c -= self.fhe_random_sub_s

        # client
        self.random_sub_c = self.fhe_builder.decrypt_to_torch(self.fhe_random_sub_c)


    def online(self, x_c, x=None):
        with NamedTimerInstance("ReluMax Online"):
            y_sub_x_c = pmod(x_c + 2 * self.q_23, self.q_23)

            self.dgk.online(y_sub_x_c)
            if x is not None:
                self.dgk.check_correctness(torch.zeros_like(x), x)
            self.fhe_enc_x = self.fhe_builder.build_enc_from_torch(x_c)

            google_vm_simulator.simulate(self.fhe_enc_x)

            # server
            with NamedTimerInstance("ReluMax Server Online"):
                self.fhe_enc_x += self.fhe_x_s
                self.fhe_plain_dgk_x_leq_y_s = self.fhe_builder.build_plain_from_torch(self.dgk.dgk_x_leq_y_s)
                self.fhe_multiplied = self.fhe_enc_x
                self.fhe_multiplied *= self.fhe_plain_dgk_x_leq_y_s
                self.fhe_multiplied += self.fhe_multipled_mask

            google_vm_simulator.simulate(self.fhe_multiplied)

            # client
            with NamedTimerInstance("DgkMax Client Online"):
                self.multipled = self.fhe_builder.decrypt_to_torch(self.fhe_multiplied)
                self.random_unmask = pmod(
                    x_c.double() * self.dgk.dgk_x_leq_y_c.double() + self.random_sub_c,
                    self.q_23)
                self.max_c = pmod(self.multipled + self.random_unmask, self.q_23)

    def check_correctness(self, x, extra=None):
        x = torch.where(x < self.q_23 // 2, x, x - self.q_23).cuda()
        self.max_expected = pmod(torch.max(x, torch.zeros_like(x)) + self.q_23, self.q_23)
        self.max_recon = pmod(self.max_s.double() + self.max_c.double() + 2 * self.q_23, self.q_23)
        print("ReluDgkSingleThread")
        compare_expected_actual(self.max_expected, self.max_recon, verbose=True, get_relative=True)

        if extra is not None:
            print("ReluDgkSingleThread Recon")
            extra = pmod(extra + 2 * self.q_23, self.q_23)
            compare_expected_actual(self.max_expected, extra.cuda(), verbose=True, get_relative=True)


def dgk_relu_simulation():
    print()
    print("Single Thread Simulation for Relu")
    data_bit = 17
    work_bit = 17
    data_range = 2 ** data_bit
    q_23 = 786433
    q_16 = 12289
    num_elem = 2 ** 15

    print(f"Number of element: {num_elem}")

    relu_dgk = ReluDgkSingleThread(num_elem, q_23, q_16, work_bit, data_bit)
    prev_swap_prot = SwapToServerOffline(num_elem, q_23)
    next_swap_prot = SwapToClientOffline(num_elem, q_23)

    prev_c = gen_unirand_int_grain(0, q_23 - 1, num_elem).cuda()
    next_c = gen_unirand_int_grain(0, q_23 - 1, num_elem).cuda()
    img_s = gen_unirand_int_grain(0, q_23 - 1, num_elem).cuda()
    prev_swap_prot.offline(prev_c, output_s=img_s)
    relu_dgk.offline(x_s=img_s)
    next_swap_prot.offline(relu_dgk.max_s, output_c=next_c)

    img = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem).cuda()
    with NamedTimerInstance("dgk_relu_simulation online"):
        img_prev_s = pmod(img - prev_c + 2 * q_23, q_23)
        prev_swap_prot.online(img_prev_s)
        img_c = prev_swap_prot.output_c.cuda()
        relu_dgk.online(img_c)
        next_swap_prot.online(relu_dgk.max_c.cpu().float())
        next_s = next_swap_prot.output_s

    relu_dgk.check_correctness(img, extra=next_s.cuda() + next_c.cuda())

    print("totally transferred MB:", google_vm_simulator.total_bit / (2 ** 20 * 8))
    print(f"totally transfer time: {google_vm_simulator.total_time * 10**3} ms")
    google_vm_simulator.reset()


class ReluDgkSharesMultSingleThread(DgkBase):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit):
        super(ReluDgkSharesMultSingleThread, self).__init__(num_elem, q_23, q_16, work_bit, data_bit)
        self.dgk = DgkSingleThread(num_elem, q_23, q_16, work_bit, data_bit)
        self.shares_mult = SharesMultSingleThread(num_elem, q_23)
        self.fhe_builder = FheBuilder(q_23, 4096)
        self.fhe_builder.generate_keys()
        self.is_server_input_offline = False

    def offline(self, x_s=None):
        # both
        if x_s is not None:
            self.y_sub_x_s = x_s
            self.is_server_input_offline = True
        else:
            self.y_sub_x_s = None

        self.dgk.offline(y_sub_x_s=self.y_sub_x_s)
        self.shares_mult.offline()

    def online(self, x_c, x_s=None, x=None):
        with NamedTimerInstance("ReluMax Online"):
            # server
            if x_s is not None and self.is_server_input_offline:
                raise Exception("Do not input server shares in both offline and online phase.")
            if x_s is not None:
                self.x_s = x_s
                self.y_sub_x_s = x_s

            self.x_c = x_c
            self.y_sub_x_c = pmod(x_c, self.q_23)

            self.dgk.online(self.y_sub_x_c, y_sub_x_s=None if self.is_server_input_offline else self.y_sub_x_s)
            self.shares_mult.online(self.x_s, self.x_c, self.dgk.dgk_x_leq_y_s, self.dgk.dgk_x_leq_y_c)
            if x is not None:
                self.dgk.check_correctness(torch.zeros_like(x), x)

            self.max_s = pmod(self.shares_mult.c_s, self.q_23)
            self.max_c = pmod(self.shares_mult.c_c, self.q_23)

    def check_correctness(self, x, extra=None):
        x = torch.where(x < self.q_23 // 2, x, x - self.q_23).cuda()
        self.max_expected = pmod(torch.max(x, torch.zeros_like(x)) + self.q_23, self.q_23)
        self.max_recon = pmod(self.max_s.double() + self.max_c.double() + 2 * self.q_23, self.q_23)
        print("ReluDgkSingleThread")
        compare_expected_actual(self.max_expected, self.max_recon, verbose=True, get_relative=True)

        if extra is not None:
            print("ReluDgkSingleThread Recon")
            extra = pmod(extra + 2 * self.q_23, self.q_23)
            compare_expected_actual(self.max_expected, extra.cuda(), verbose=True, get_relative=True)


def dgk_relu_shares_mult_simulation():
    print()
    print("Single Thread Simulation for Relu with Shares Mult")
    data_bit = 17
    work_bit = 17
    data_range = 2 ** data_bit
    q_23 = 786433
    q_16 = 12289
    num_elem = 2 ** 17

    print(f"Number of element: {num_elem}")

    relu_dgk = ReluDgkSharesMultSingleThread(num_elem, q_23, q_16, work_bit, data_bit)
    next_swap_prot = SwapToClientOffline(num_elem, q_23)

    next_c = gen_unirand_int_grain(0, q_23 - 1, num_elem).cuda()
    img_c = gen_unirand_int_grain(0, q_23 - 1, num_elem).cuda()
    relu_dgk.offline()
    next_swap_prot.offline(output_c=next_c)

    img = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem).cuda()
    with NamedTimerInstance("dgk_relu_simulation online"):
        img_s = pmod(img - img_c, q_23)
        relu_dgk.online(img_c, x_s=img_s)
        next_swap_prot.online(relu_dgk.max_c.cpu().float(), input_s=relu_dgk.max_s.cpu().float())
        next_s = next_swap_prot.output_s

    relu_dgk.check_correctness(img, extra=next_s.cuda() + next_c.cuda())

    print("totally transferred MB:", google_vm_simulator.total_bit / (2 ** 20 * 8))
    print(f"totally transfer time: {google_vm_simulator.total_time * 10**3} ms")
    google_vm_simulator.reset()

class Maxpool2x2DgkSingleThread(DgkBase):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit, img_hw):
        super(Maxpool2x2DgkSingleThread, self).__init__(num_elem, q_23, q_16, work_bit, data_bit)
        self.img_hw = img_hw
        if num_elem % 4 != 0:
            raise Exception(f"num_elem should be divisible by 4, but got {num_elem}")
        if img_hw % 2 != 0:
            raise Exception(f"img_hw should be divisible by 2, but got {img_hw}")
        self.max_dgk_1 = MaxDgkSingleThread(num_elem//2, q_23, q_16, work_bit, data_bit)
        self.max_dgk_2 = MaxDgkSingleThread(num_elem//4, q_23, q_16, work_bit, data_bit)

    def reorder_1_x(self, x: torch.Tensor) -> torch.Tensor:
        return x[::2] if x is not None else x

    def reorder_1_y(self, x: torch.Tensor) -> torch.Tensor:
        return x[1::2] if x is not None else x

    def reorder_2_x(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape([-1, self.img_hw//2])[::2, :].reshape(-1) if x is not None else x

    def reorder_2_y(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape([-1, self.img_hw//2])[1::2, :].reshape(-1) if x is not None else x

    def offline(self, img_s=None):
        self.max_dgk_1.offline(x_s=self.reorder_1_x(img_s), y_s=self.reorder_1_y(img_s))
        img_2_s = self.max_dgk_1.max_s
        self.max_dgk_2.offline(x_s=self.reorder_2_x(img_2_s), y_s=self.reorder_2_y(img_2_s))
        self.max_s = self.max_dgk_2.max_s

    def online(self, img_c, img=None):
        with NamedTimerInstance("Maxpool2x2 Dgk Online"):
            img_1_x = self.reorder_1_x(img)
            img_1_y = self.reorder_1_y(img)
            img_2 = torch.max(img_1_x, img_1_y) if img is not None else None
            img_2_x = self.reorder_2_x(img_2)
            img_2_y = self.reorder_2_y(img_2)
            self.max_dgk_1.online(self.reorder_1_x(img_c), self.reorder_1_y(img_c), x=img_1_x, y=img_1_y)
            img_2_c = self.max_dgk_1.max_c.float()
            img_2_x_c = self.reorder_2_x(img_2_c)
            img_2_y_c = self.reorder_2_y(img_2_c)
            self.max_dgk_2.online(img_2_x_c, img_2_y_c, x=img_2_x, y=img_2_y)
            self.max_c = self.max_dgk_2.max_c

        if img is not None:
            self.check_correctness(img)

    def check_correctness(self, img, extra=None):
        pool = torch.nn.MaxPool2d(2)
        img = torch.where(img < self.q_23 // 2, img, img - self.q_23).cuda()
        output_expected = pool(img.reshape([-1, self.img_hw, self.img_hw])).reshape(-1)
        self.max_expected = pmod(output_expected + self.q_23, self.q_23)
        self.max_recon = pmod(self.max_c.double() + self.max_s.double() + 2 * self.q_23, self.q_23)
        print("Maxpool2x2DgkSingleThread")
        compare_expected_actual(self.max_expected, self.max_recon, verbose=True, get_relative=True)

        if extra is not None:
            print("Maxpool2x2DgkSingleThread Recon")
            extra = pmod(extra + 2 * self.q_23, self.q_23)
            compare_expected_actual(self.max_expected, extra.cuda(), verbose=True, get_relative=True)


def dgk_maxpool2x2_simulation():
    print()
    print("Single Thread Simulation for Maxpool2x2")
    data_bit = 17
    work_bit = 17
    data_range = 2 ** data_bit
    q_23 = 786433
    q_16 = 12289
    num_elem = 2 ** 15
    img_hw = 32

    print(f"Number of element: {num_elem}")

    maxpool2x2_dgk = Maxpool2x2DgkSingleThread(num_elem, q_23, q_16, work_bit, data_bit, img_hw)
    prev_swap_prot = SwapToServerOffline(num_elem, q_23)
    next_swap_prot = SwapToClientOffline(num_elem//4, q_23)

    prev_c = gen_unirand_int_grain(0, q_23 - 1, num_elem).cuda()
    next_c = gen_unirand_int_grain(0, q_23 - 1, num_elem//4).cuda()
    img_s = gen_unirand_int_grain(0, q_23 - 1, num_elem).cuda()
    prev_swap_prot.offline(prev_c, output_s=img_s)
    maxpool2x2_dgk.offline(img_s=img_s)
    next_swap_prot.offline(maxpool2x2_dgk.max_s, output_c=next_c)

    # img = torch.arange(num_elem).float().cuda()
    img = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem).cuda()
    with NamedTimerInstance("dgk_maxpool2x2_simulation online"):
        img_prev_s = pmod(img - prev_c + 2 * q_23, q_23)
        prev_swap_prot.online(img_prev_s)
        img_c = prev_swap_prot.output_c.cuda()
        maxpool2x2_dgk.online(img_c)
        next_swap_prot.online(maxpool2x2_dgk.max_c.cpu().float())
        next_s = next_swap_prot.output_s

    maxpool2x2_dgk.check_correctness(img, extra=next_s.cuda() + next_c.cuda())

    print("totally transferred MB:", google_vm_simulator.total_bit / (2 ** 20 * 8))
    print(f"totally transfer time: {google_vm_simulator.total_time * 10**3} ms")
    google_vm_simulator.reset()


class Maxpool2x2DgkSharesMultSingleThread(DgkBase):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit, img_hw):
        super(Maxpool2x2DgkSharesMultSingleThread, self).__init__(num_elem, q_23, q_16, work_bit, data_bit)
        self.img_hw = img_hw
        if num_elem % 4 != 0:
            raise Exception(f"num_elem should be divisible by 4, but got {num_elem}")
        if img_hw % 2 != 0:
            raise Exception(f"img_hw should be divisible by 2, but got {img_hw}")
        self.max_dgk_1 = MaxDgkSharesMultSingleThread(num_elem//2, q_23, q_16, work_bit, data_bit)
        self.max_dgk_2 = MaxDgkSharesMultSingleThread(num_elem//4, q_23, q_16, work_bit, data_bit)

    def reorder_1_x(self, x: torch.Tensor) -> torch.Tensor:
        return x[::2] if x is not None else x

    def reorder_1_y(self, x: torch.Tensor) -> torch.Tensor:
        return x[1::2] if x is not None else x

    def reorder_2_x(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape([-1, self.img_hw//2])[::2, :].reshape(-1) if x is not None else x

    def reorder_2_y(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape([-1, self.img_hw//2])[1::2, :].reshape(-1) if x is not None else x

    def offline(self):
        self.max_dgk_1.offline()
        self.max_dgk_2.offline()

    def online(self, img_s, img_c, img=None):
        with NamedTimerInstance("Maxpool2x2 Dgk Online"):
            img_1_x = self.reorder_1_x(img)
            img_1_y = self.reorder_1_y(img)
            img_2 = torch.max(img_1_x, img_1_y) if img is not None else None
            img_2_x = self.reorder_2_x(img_2)
            img_2_y = self.reorder_2_y(img_2)

            img_1_x_c = self.reorder_1_x(img_c)
            img_1_y_c = self.reorder_1_y(img_c)
            img_1_x_s = self.reorder_1_x(img_s)
            img_1_y_s = self.reorder_1_y(img_s)

            self.max_dgk_1.online(img_1_x_c, img_1_y_c, x_s=img_1_x_s, y_s=img_1_y_s, x=img_1_x, y=img_1_y)

            img_2_c = self.max_dgk_1.max_c.float()
            img_2_x_c = self.reorder_2_x(img_2_c)
            img_2_y_c = self.reorder_2_y(img_2_c)
            img_2_s = self.max_dgk_1.max_s.float()
            img_2_x_s = self.reorder_2_x(img_2_s)
            img_2_y_s = self.reorder_2_y(img_2_s)

            self.max_dgk_2.online(img_2_x_c, img_2_y_c, x_s=img_2_x_s, y_s=img_2_y_s, x=img_2_x, y=img_2_y)

            self.max_s = self.max_dgk_2.max_s
            self.max_c = self.max_dgk_2.max_c

        if img is not None:
            self.check_correctness(img)

    def check_correctness(self, img, extra=None):
        pool = torch.nn.MaxPool2d(2)
        img = torch.where(img < self.q_23 // 2, img, img - self.q_23).cuda()
        output_expected = pool(img.reshape([-1, self.img_hw, self.img_hw])).reshape(-1)
        self.max_expected = pmod(output_expected + self.q_23, self.q_23)
        self.max_recon = pmod(self.max_c.double() + self.max_s.double() + 2 * self.q_23, self.q_23)
        print("Maxpool2x2DgkSingleThread")
        compare_expected_actual(self.max_expected, self.max_recon, verbose=True, get_relative=True)

        if extra is not None:
            print("Maxpool2x2DgkSingleThread Recon")
            extra = pmod(extra, self.q_23)
            compare_expected_actual(self.max_expected, extra.cuda(), verbose=True, get_relative=True)


def dgk_maxpool2x2_shares_mult_simulation():
    print()
    print("Single Thread Simulation for Maxpool2x2")
    data_bit = 17
    work_bit = 17
    data_range = 2 ** data_bit
    q_23 = 786433
    q_16 = 12289
    num_elem = 2 ** 15
    img_hw = 32

    print(f"Number of element: {num_elem}")

    maxpool2x2_dgk = Maxpool2x2DgkSharesMultSingleThread(num_elem, q_23, q_16, work_bit, data_bit, img_hw)
    next_swap_prot = SwapToClientOffline(num_elem//4, q_23)

    next_c = gen_unirand_int_grain(0, q_23 - 1, num_elem//4).cuda()
    maxpool2x2_dgk.offline()
    next_swap_prot.offline(output_c=next_c)

    # img = torch.arange(num_elem).float().cuda()
    img = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem).cuda()
    img_s = gen_unirand_int_grain(0, q_23 - 1, num_elem).cuda()
    img_c = pmod(img - img_s, q_23)
    with NamedTimerInstance("dgk_maxpool2x2_simulation online"):
        maxpool2x2_dgk.online(img_s, img_c)
        next_swap_prot.online(maxpool2x2_dgk.max_c.cpu().float(), input_s=maxpool2x2_dgk.max_s.cpu().float())
        next_s = next_swap_prot.output_s
        next_c = next_swap_prot.output_c

    maxpool2x2_dgk.check_correctness(img, extra=pmod(next_s.cuda() + next_c.cuda(), q_23))

    print("totally transferred MB:", google_vm_simulator.total_bit / (2 ** 20 * 8))
    print(f"totally transfer time: {google_vm_simulator.total_time * 10**3} ms")
    google_vm_simulator.reset()


class TruncSingleThread(DgkBase):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit, divisor_pow):
        super().__init__(num_elem, q_23, q_16, work_bit, data_bit)
        self.fhe_builder = FheBuilder(q_23, Config.n_23)
        self.fhe_builder.generate_keys()
        self.divisor_pow = divisor_pow
        self.div = 2 ** divisor_pow
        self.nullify_threshold = (q_23 - 1) / 2
        assert(isinstance(self.divisor_pow, int))
        assert(1 <= self.divisor_pow <= int(floor(log2(q_23))))

    def generate_mask(self):
        return gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).cuda()

    def mod_to_modulus(self, x):
        return pmod(x, self.q_23)

    def offline(self):
        # client
        self.pre_c = self.generate_mask()
        fhe_pre_c = self.fhe_builder.build_enc_from_torch(self.pre_c)

        # server
        self.elem_zeros = torch.zeros(self.num_elem).cuda()
        self.r = self.generate_mask()
        self.mult = torch.where((self.r < self.nullify_threshold),
                                self.elem_zeros, self.elem_zeros + self.q_23//self.div).double()
        self.wrap_mask_s = self.generate_mask()
        fhe_mult = self.fhe_builder.build_plain_from_torch(self.mult)
        fhe_bias = self.fhe_builder.build_plain_from_torch(self.wrap_mask_s)
        fhe_pre_c *= fhe_mult
        fhe_pre_c += fhe_bias

        # client
        self.fhe_builder.noise_budget(fhe_pre_c, name="fhe_pre_c")
        self.wrap_c = self.fhe_builder.decrypt_to_torch(fhe_pre_c)

    def online(self, x_s, x_c):
        # server
        sum_xr_s = self.mod_to_modulus(self.r + x_s)
        google_vm_simulator.simulate(sum_xr_s.type(torch.float).cpu().cuda())

        # client
        sum_xr = self.mod_to_modulus(sum_xr_s + x_c)
        pre_s = torch.where(sum_xr < self.nullify_threshold, self.elem_zeros + 1, self.elem_zeros)
        pre_s = self.mod_to_modulus(pre_s - self.pre_c)
        self.out_c = self.mod_to_modulus(sum_xr//self.div + self.wrap_c)
        google_vm_simulator.simulate(pre_s.type(torch.float).cpu().cuda())

        # server
        self.wrap_s = self.mult * pre_s - self.wrap_mask_s
        self.out_s = self.mod_to_modulus(-self.r//self.div + self.wrap_s)


def trunc_simulation():
    print()
    print("Single Thread Simulation for trunc")
    data_bit = 17
    work_bit = 17
    data_range = 2 ** data_bit
    q_23 = 786433
    q_16 = 12289
    num_elem = 2 ** 17
    pow_to_div = 9

    print(f"Number of element: {num_elem}")
    
    def check_correctness(out_s, out_c, img, pow_to_div):
        div = 2 ** pow_to_div
        expected = pmod(img//div, q_23)
        actual = pmod(out_s + out_c, q_23)
        err_indices = ((actual - expected) != 0) & ((actual - expected) != 1)
        unacceptable = torch.sum(err_indices).item()
        print("unacceptable:", unacceptable)

    prot = TruncSingleThread(num_elem, q_23, q_16, work_bit, data_bit, pow_to_div)

    img = gen_unirand_int_grain(0, data_range // 2, num_elem).cuda()
    img_s = gen_unirand_int_grain(0, q_23 - 1, num_elem).cuda()
    img_c = pmod(img - img_s, q_23)
    prot.offline()

    with NamedTimerInstance("dgk_relu_simulation online"):
        prot.online(img_s, img_c)

    check_correctness(prot.out_s, prot.out_c, img, pow_to_div)

    print("totally transferred MB:", google_vm_simulator.total_bit / (2 ** 20 * 8))
    print(f"totally transfer time: {google_vm_simulator.total_time * 10**3} ms")
    google_vm_simulator.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t",
                        type=str,
                        default="all",
                        help="The test specified to run")
    parser.add_argument("--repeat", "-r",
                        type=int,
                        default=1,
                        help="The repetition times of the test")
    args = parser.parse_args()
    run_test = args.test
    num_repeat = args.repeat

    warming_up_cuda()

    for i in range(num_repeat):
        if run_test in ["all", "refresh"]:
            test_fhe_refresh()
        if run_test in ["all", "swap_server"]:
            swap_to_server_offline_simulation()
        if run_test in ["all", "swap_client"]:
            swap_to_client_offline_simulation()
        if run_test in ["all", "shares_mult"]:
            shares_mult_simulation()
        if run_test in ["all", "dgk"]:
            dgk_simulation()
        if run_test in ["all", "max"]:
            dgk_max_simulation()
        if run_test in ["all", "max_shares"]:
            dgk_max_shares_mults_simulation()
        if run_test in ["all", "relu"]:
            dgk_relu_simulation()
        if run_test in ["all", "relu_shares"]:
            dgk_relu_shares_mult_simulation()
        if run_test in ["all", "maxpool"]:
            dgk_maxpool2x2_simulation()
        if run_test in ["all", "maxpool_shares"]:
            dgk_maxpool2x2_shares_mult_simulation()
        if run_test in ["all", "trunc"]:
            trunc_simulation()
