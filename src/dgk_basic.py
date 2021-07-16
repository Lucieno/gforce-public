import sys
from itertools import product

import torch

from comm import CommBase, CommFheBuilder, BlobFheEnc, BlobTorch, BlobFheEnc2D, init_communicate, end_communicate, \
    torch_sync, TrafficRecord
from config import Config
from dgk_single_thread import DgkBase
from enc_refresher import EncRefresherServer, EncRefresherClient
from fhe import FheBuilder
from logger_utils import Logger
from timer_utils import NamedTimerInstance
from torch_utils import gen_unirand_int_grain, pmod, shuffle_torch, compare_expected_actual, marshal_funcs, \
    argparser_distributed, warming_up_cuda


class DgkCommBase(DgkBase):
    modulus = None
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str, rank: int, class_name: str):
        super(DgkCommBase, self).__init__(num_elem, q_23, q_16, work_bit, data_bit)
        self.class_name = class_name
        self.name = name
        self.rank = rank
        self.fhe_builder_16 = fhe_builder_16
        self.fhe_builder_23 = fhe_builder_23
        self.comm_base = CommBase(rank, name)
        self.comm_fhe_16 = CommFheBuilder(rank, fhe_builder_16, self.sub_name("comm_fhe_16"))
        self.comm_fhe_23 = CommFheBuilder(rank, fhe_builder_23, self.sub_name("comm_fhe_23"))

        assert(self.fhe_builder_16.modulus == self.q_16)
        assert(self.fhe_builder_23.modulus == self.q_23)

    def generate_random(self):
        return gen_unirand_int_grain(0, self.modulus - 1, self.num_elem)

    def mod_to_modulus(self, input):
        return pmod(input, self.modulus)

    def sub_name(self, sub_name: str) -> str:
        return self.name + '_' + self.class_name +'_' + sub_name


class DgkBitCommon(DgkBase):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit, name: str,
                 comm_base: CommBase, comm_fhe_16: CommFheBuilder, comm_fhe_23: CommFheBuilder):
        super(DgkBitCommon, self).__init__(num_elem, q_23, q_16, work_bit, data_bit)
        self.comm_base = comm_base
        self.comm_fhe_16 = comm_fhe_16
        self.comm_fhe_23 = comm_fhe_23
        self.name = name

        self.beta_i_c = BlobFheEnc2D(self.decomp_bit_shape, comm_fhe_16, self.sub_name("beta_i_c"))
        self.delta_b_c = BlobFheEnc(self.num_elem, comm_fhe_23, self.sub_name("delta_b_c"))
        self.z_work_c = BlobFheEnc(self.num_elem, comm_fhe_23, self.sub_name("z_work_c"))
        self.c_i_c = BlobFheEnc2D(self.sum_shape, comm_fhe_16, self.sub_name("c_i_c"))
        self.dgk_x_leq_y_c = BlobFheEnc(self.num_elem, comm_fhe_23, self.sub_name("dgk_x_leq_y_c"))
        self.delta_xor_c = BlobFheEnc(self.num_elem, comm_fhe_23, self.sub_name("delta_xor_c"))
        self.fhe_pre_corr_mod = BlobFheEnc(self.num_elem, comm_fhe_23, self.sub_name("fhe_pre_corr_mod"))
        self.fhe_corr_mod_c = BlobFheEnc(self.num_elem, comm_fhe_23, self.sub_name("fhe_corr_mod_c"))

        self.z_s = BlobTorch(self.num_elem, torch.float, self.comm_base, self.sub_name("z_s"))
        # self.beta_i_s = BlobTorch(self.decomp_bit_shape, torch.int16, self.comm_base, self.sub_name("beta_i_s"), comp_dtype=torch.float)
        self.beta_i_s = BlobTorch(self.decomp_bit_shape, torch.float, self.comm_base, self.sub_name("beta_i_s"), comp_dtype=torch.float)
        # self.c_i_s = BlobTorch(self.sum_shape, torch.int16, self.comm_base, self.sub_name("c_i_s"), comp_dtype=torch.float)
        self.c_i_s = BlobTorch(self.sum_shape, torch.float, self.comm_base, self.sub_name("c_i_s"), comp_dtype=torch.float)
        self.delta_b_s = BlobTorch(self.num_elem, torch.float, self.comm_base, self.sub_name("delta_b_s"))
        self.z_work_s = BlobTorch(self.num_elem, torch.float, self.comm_base, self.sub_name("z_work_s"))
        self.pre_corr_mod_s = BlobTorch(self.num_elem, torch.float, self.comm_base, self.sub_name("pre_corr_mod_s"))

        self.offline_server_send = [self.c_i_c, self.dgk_x_leq_y_c, self.delta_xor_c, self.fhe_corr_mod_c]
        self.offline_client_send = [self.beta_i_c, self.delta_b_c, self.z_work_c, self.fhe_pre_corr_mod]
        self.online_server_send = [self.z_s, self.c_i_s]
        self.online_client_send = [self.beta_i_s, self.delta_b_s, self.z_work_s, self.pre_corr_mod_s]

    def sub_name(self, sub_name: str) -> str:
        return self.name + '_DgkBit_' + sub_name

    def decomp_to_bit(self, x, res=None):
        tmp_x = torch.clone(x).to(Config.device)
        res = torch.zeros([self.work_bit, self.num_elem]) if res is None else res
        for i in range(self.work_bit):
            res[i] = pmod(tmp_x, 2)
            tmp_x //= 2
        return res


class DgkBitServer(DgkBase):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str, is_shuffle=None):
        super(DgkBitServer, self).__init__(num_elem, q_23, q_16, work_bit, data_bit, name=name)
        self.fhe_builder_16 = fhe_builder_16
        self.fhe_builder_23 = fhe_builder_23
        self.comm_base = CommBase(Config.server_rank, name)
        self.comm_fhe_16 = CommFheBuilder(Config.server_rank, self.fhe_builder_16, name+'_'+"comm_fhe_16")
        self.comm_fhe_23 = CommFheBuilder(Config.server_rank, self.fhe_builder_23, name+'_'+"comm_fhe_23")
        self.common = DgkBitCommon(num_elem, q_23, q_16, work_bit, data_bit, name,
                                   self.comm_base, self.comm_fhe_16, self.comm_fhe_23)
        self.is_shuffle = Config.is_shuffle if is_shuffle is None else is_shuffle

    def xor_fhe(self, alpha_i, fhe_enc, mask_s, modulus, change_sign):
        if modulus == self.q_16:
            fhe_builder = self.fhe_builder_16
        elif modulus == self.q_23:
            fhe_builder = self.fhe_builder_23
        else:
            raise Exception(f"Unknown modulus: {modulus}")
        zeros = torch.zeros_like(alpha_i)
        mult = torch.where(alpha_i == change_sign, modulus - 1 + zeros, 1 + zeros)
        bias = torch.where(alpha_i == change_sign, 1 + zeros, zeros)
        fhe_mult = fhe_builder.build_plain_from_torch(mult)
        fhe_bias = fhe_builder.build_plain_from_torch(bias)
        fhe_mask_s = fhe_builder.build_plain_from_torch(mask_s)
        fhe_enc *= fhe_mult
        fhe_enc += fhe_bias
        fhe_enc += fhe_mask_s
        del fhe_mult, fhe_bias, fhe_mask_s
        return fhe_enc

    def xor_alpha_known_offline(self, alpha_i, fhe_beta_i_c, mask_s):
        assert(len(alpha_i) == self.work_bit)
        assert(len(fhe_beta_i_c) == self.work_bit)
        assert(len(mask_s) == self.work_bit)
        return [self.xor_fhe(alpha_i[i], fhe_beta_i_c[i], mask_s[i], self.q_16, 1) for i in range(self.work_bit)]

    def xor_alpha_known_online(self, alpha_i, beta_i_s, mask_s, modulus):
        res = torch.where(alpha_i == 1, -beta_i_s, beta_i_s)
        res += modulus - mask_s
        res.fmod_(modulus)
        return res

    def generate_fhe_shuffled(self, shuffle_order, enc):
        num_batch = self.num_work_batch
        fhe_builder = self.fhe_builder_16
        # res = [fhe_builder.build_enc(self.num_elem) for i in range(num_batch)]
        res = [None for i in range(num_batch)]
        zeros = torch.tensor(0).type(torch.int64)
        shuffle_order = shuffle_order.cpu()
        for dst, src in product(range(num_batch), range(num_batch)):
            mask = torch.where(shuffle_order[src, :] == dst, zeros + 1, zeros)
            # print(torch.sum(mask))
            fhe_mask = fhe_builder.build_plain_from_torch(mask)
            enc_tmp = enc[src].copy()
            # fhe_builder.noise_budget(enc_tmp, "enc_tmp")
            enc_tmp *= fhe_mask
            # fhe_builder.noise_budget(enc_tmp, "enc_tmp")
            if src == 0:
                res[dst] = enc_tmp
            else:
                res[dst] += enc_tmp
        return res

    def sum_c_i_offline(self, delta_a, fhe_beta_i_c, fhe_alpha_beta_xor_c, s, alpha_i,
                        ci_mask_s, mult_mask_s, shuffle_order):
        # the last row of sum_xor is c_{-1}, which helps check the case with x == y
        fhe_builder = self.fhe_builder_16
        # fhe_sum_xor = [fhe_builder.build_enc(self.num_elem) for i in range(self.num_work_batch)]
        fhe_sum_xor = [None for i in range(self.num_work_batch)]
        fhe_sum_xor[self.work_bit - 1] = fhe_builder.build_enc(self.num_elem)
        for i in range(self.work_bit - 1)[::-1]:
            fhe_sum_xor[i] = fhe_sum_xor[i + 1].copy()
            fhe_sum_xor[i] += fhe_alpha_beta_xor_c[i + 1]
        fhe_delta_a = fhe_builder.build_plain_from_torch(delta_a)
        fhe_sum_xor[self.work_bit] = fhe_sum_xor[0].copy()
        fhe_sum_xor[self.work_bit] += fhe_alpha_beta_xor_c[0]
        fhe_sum_xor[self.work_bit] += fhe_delta_a
        del fhe_delta_a

        for i in range(self.work_bit)[::-1]:
            fhe_mult_3 = fhe_builder.build_plain_from_torch(pmod(3 * mult_mask_s[i].cpu(), self.q_16))
            fhe_mult_mask_s = fhe_builder.build_plain_from_torch(mult_mask_s[i])
            masked_s = pmod(s.type(torch.int64) * mult_mask_s[i].type(torch.int64), self.q_16).type(torch.float32)
            # print("s * mult_mask_s[i]", torch.max(masked_s))
            fhe_s = fhe_builder.build_plain_from_torch(masked_s)
            fhe_alpha_i = fhe_builder.build_plain_from_torch(alpha_i[i] * mult_mask_s[i])
            fhe_ci_mask_s = fhe_builder.build_plain_from_torch(ci_mask_s[i])
            fhe_beta_i_c[i] *= fhe_mult_mask_s
            fhe_sum_xor[i] *= fhe_mult_3
            fhe_sum_xor[i] -= fhe_beta_i_c[i]
            fhe_sum_xor[i] += fhe_s
            fhe_sum_xor[i] += fhe_alpha_i
            fhe_sum_xor[i] += fhe_ci_mask_s

            del fhe_mult_3, fhe_mult_mask_s, fhe_s, fhe_alpha_i, fhe_ci_mask_s

        fhe_mult_mask_s = fhe_builder.build_plain_from_torch(mult_mask_s[self.work_bit])
        fhe_ci_mask_s = fhe_builder.build_plain_from_torch(ci_mask_s[self.work_bit])
        fhe_sum_xor[self.work_bit] *= fhe_mult_mask_s
        fhe_sum_xor[self.work_bit] += fhe_ci_mask_s

        del fhe_mult_mask_s, fhe_ci_mask_s

        if self.is_shuffle:
            with NamedTimerInstance("Shuffle"):
                refresher = EncRefresherServer(self.sum_shape, fhe_builder, self.sub_name("shuffle_refresher"))
                with NamedTimerInstance("refresh"):
                    new_fhe_sum_xor = refresher.request(fhe_sum_xor)
                del fhe_sum_xor
                fhe_sum_xor = self.generate_fhe_shuffled(shuffle_order, new_fhe_sum_xor)
                del refresher

        return fhe_sum_xor

    def sum_c_i_common(self, alpha_beta_xor_share):
        sum_xor = torch.zeros(self.sum_shape).to(Config.device)
        # the last row of sum_xor is c_{-1}, which helps check the case with x == y
        for i in range(self.work_bit - 1)[::-1]:
            sum_xor[i] = sum_xor[i + 1] + alpha_beta_xor_share[i + 1]
        return sum_xor

    def sum_c_i_online(self, beta_i_s, alpha_beta_xor_s, ci_mask_s, mult_mask_s, shuffle_order):
        sum_xor = self.fast_zeros_sum_xor
        # the last row of sum_xor is c_{-1}, which helps check the case with x == y
        for i in range(self.work_bit - 1)[::-1]:
            sum_xor[i] = sum_xor[i + 1] + alpha_beta_xor_s[i + 1]
        sum_xor[self.work_bit] = sum_xor[0] + alpha_beta_xor_s[0]
        for i in range(self.work_bit)[::-1]:
            sum_xor[i] = 3 * sum_xor[i] - beta_i_s[i]
        sum_xor = sum_xor.double() * mult_mask_s
        sum_xor -= ci_mask_s
        sum_xor = pmod(sum_xor, self.q_16).float().to(Config.device)
        if self.is_shuffle:
            sum_xor = shuffle_torch(sum_xor, shuffle_order)
        return sum_xor

    def xor_delta_known_offline(self, alpha_i, fhe_beta_i_c, mask_s):
        return self.xor_fhe(alpha_i, fhe_beta_i_c, mask_s, self.q_23, 0)

    def xor_delta_known_online(self, alpha_i, beta_i_s, mask_s, modulus):
        res = torch.where(alpha_i == 1, beta_i_s, -beta_i_s)
        res += modulus - mask_s
        res.fmod_(modulus)
        return res

    def mod_div_offline(self):
        fhe_builder = self.fhe_builder_23

        self.elem_zeros = torch.zeros(self.num_elem).to(Config.device)
        self.correct_mod_div_work_mult = torch.where((self.r < self.nullify_threshold),
                                                     self.elem_zeros,
                                                     self.elem_zeros + self.q_23 // self.work_range).double()
        self.correct_mod_div_work_mask_s = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).to(Config.device)
        fhe_mult = fhe_builder.build_plain_from_torch(self.correct_mod_div_work_mult)
        fhe_bias = fhe_builder.build_plain_from_torch(self.correct_mod_div_work_mask_s)
        fhe_correct_mod_div_work = self.common.fhe_pre_corr_mod.get_recv()
        fhe_correct_mod_div_work *= fhe_mult
        fhe_correct_mod_div_work += fhe_bias
        del fhe_mult, fhe_bias

        self.common.fhe_corr_mod_c.send(fhe_correct_mod_div_work)

    def mod_div_online(self):
        pre_correct_mod_div_s = self.common.pre_corr_mod_s.get_recv()

        self.correct_mod_div_work_s = pmod(
            self.correct_mod_div_work_mult * pre_correct_mod_div_s - self.correct_mod_div_work_mask_s, self.q_23)

    def offline_recv(self):
        for blob in self.common.offline_client_send:
            blob.prepare_recv()

    def online_recv(self):
        for blob in self.common.online_client_send:
            blob.prepare_recv()

    def offline(self):
        self.offline_recv()

        self.delta_a = gen_unirand_int_grain(0, 1, self.num_elem).to(Config.device)
        # self.s = pmod(1 - 2 * self.delta_a, self.q_16)
        self.s = pmod(1 - 2 * self.delta_a, self.q_16)
        # self.r = gen_unirand_int_grain(0, 2 ** (self.work_bit + 1) - 1, self.num_elem).to(Config.device)
        self.r = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).to(Config.device)
        self.alpha = pmod(self.r, self.work_range)
        self.alpha_i = self.common.decomp_to_bit(self.alpha).to(Config.device)
        self.beta_i_mask_s = gen_unirand_int_grain(0, self.q_16 - 1, self.decomp_bit_shape).to(Config.device)
        self.ci_mask_s = gen_unirand_int_grain(0, self.q_16 - 1, [self.work_bit + 1, self.num_elem]).to(Config.device)
        self.ci_mult_mask_s = gen_unirand_int_grain(1, self.q_16 - 1, [self.work_bit + 1, self.num_elem]).to(Config.device)
        self.shuffle_order = torch.rand([self.work_bit + 1, self.num_elem]).argsort(dim=0).to(Config.device)
        self.delta_xor_mask_s = gen_unirand_int_grain(0, self.q_16 - 1, self.num_elem).to(Config.device)
        self.dgk_x_leq_y_mask_s = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).to(Config.device)
        self.fast_zeros_sum_xor = torch.zeros(self.sum_shape).to(Config.device)

        self.mod_div_offline()

        refresher_ab_xor_c = EncRefresherServer(
            self.decomp_bit_shape, self.fhe_builder_16, self.common.sub_name("refresher_ab_xor_c"))

        fhe_beta_i_c = self.common.beta_i_c.get_recv()
        fhe_beta_i_c_for_sum_c = [fhe_beta_i_c[i].copy() for i in range(len(fhe_beta_i_c))]
        fhe_alpha_beta_xor_c = self.xor_alpha_known_offline(self.alpha_i, fhe_beta_i_c, self.beta_i_mask_s)
        fhe_alpha_beta_xor_c = refresher_ab_xor_c.request(fhe_alpha_beta_xor_c)
        fhe_c_i_c = self.sum_c_i_offline(self.delta_a, fhe_beta_i_c_for_sum_c, fhe_alpha_beta_xor_c, self.s,
                                         self.alpha_i, self.ci_mask_s, self.ci_mult_mask_s, self.shuffle_order)
        self.common.c_i_c.send(fhe_c_i_c)

        fhe_delta_b_c = self.common.delta_b_c.get_recv()
        fhe_delta_xor_c = self.xor_delta_known_offline(self.delta_a, fhe_delta_b_c, self.delta_xor_mask_s)
        self.common.delta_xor_c.send(fhe_delta_xor_c)

        fhe_z_work_c = self.common.z_work_c.get_recv()
        fhe_z_work_c -= fhe_delta_xor_c
        fhe_z_work_c -= self.fhe_builder_23.build_plain_from_torch(self.dgk_x_leq_y_mask_s)
        self.common.dgk_x_leq_y_c.send(fhe_z_work_c)

        for ct in fhe_c_i_c + fhe_beta_i_c + fhe_beta_i_c_for_sum_c:
            del ct
        del fhe_beta_i_c, fhe_beta_i_c_for_sum_c, fhe_alpha_beta_xor_c, fhe_c_i_c, fhe_delta_b_c, fhe_delta_xor_c, fhe_z_work_c
        del refresher_ab_xor_c

        self.online_recv()
        torch_sync()

    def online(self, y_sub_x_s):
        self.z_s = pmod(y_sub_x_s + self.work_range + self.r, self.q_23)
        self.common.z_s.send(self.z_s)

        beta_i_s = self.common.beta_i_s.get_recv()
        alpha_beta_xor_s = self.xor_alpha_known_online(self.alpha_i, beta_i_s, self.beta_i_mask_s, self.q_16)
        c_i_s = self.sum_c_i_online(beta_i_s, alpha_beta_xor_s, self.ci_mask_s, self.ci_mult_mask_s, self.shuffle_order)
        self.common.c_i_s.send(c_i_s)


        delta_b_s = self.common.delta_b_s.get_recv()
        delta_xor_s = self.xor_delta_known_online(self.delta_a, delta_b_s, self.delta_xor_mask_s, self.q_23)
        z_work_s = self.common.z_work_s.get_recv()
        self.mod_div_online()

        self.dgk_x_leq_y_s = pmod(
            z_work_s - ((self.r // self.work_range) + delta_xor_s) + self.correct_mod_div_work_s + self.dgk_x_leq_y_mask_s,
            self.q_23)


class DgkBitClient(DgkBase):
    z: torch.Tensor
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str, is_shuffle=None):
        super(DgkBitClient, self).__init__(num_elem, q_23, q_16, work_bit, data_bit, name=name)
        self.fhe_builder_16 = fhe_builder_16
        self.fhe_builder_23 = fhe_builder_23
        self.comm_base = CommBase(Config.client_rank, name)
        self.comm_fhe_16 = CommFheBuilder(Config.client_rank, self.fhe_builder_16, name+'_'+"comm_fhe_16")
        self.comm_fhe_23 = CommFheBuilder(Config.client_rank, self.fhe_builder_23, name+'_'+"comm_fhe_23")
        self.common = DgkBitCommon(num_elem, q_23, q_16, work_bit, data_bit, name,
                                   self.comm_base, self.comm_fhe_16, self.comm_fhe_23)
        self.is_shuffle = Config.is_shuffle if is_shuffle is None else is_shuffle

    def sum_c_i_offline(self):
        if self.is_shuffle:
            # self.sum_c_refresher = EncRefresherClient(self.sum_shape, self.fhe_builder_16, self.sub_name("shuffle_refresher"))
            # self.sum_c_refresher.prepare_recv()
            self.sum_c_refresher.response()
            del self.sum_c_refresher

    def mod_div_offline(self):
        fhe_builder = self.fhe_builder_23
        self.elem_zeros = torch.zeros(self.num_elem).to(Config.device)
        self.pre_mod_div_c = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).to(Config.device)
        fhe_correct_mod_div_work = fhe_builder.build_enc_from_torch(self.pre_mod_div_c)

        self.common.fhe_pre_corr_mod.send(fhe_correct_mod_div_work)
        fhe_corr_mod_c = self.common.fhe_corr_mod_c.get_recv()
        self.correct_mod_div_work_c = fhe_builder.decrypt_to_torch(fhe_corr_mod_c)

    def mod_div_online(self, z):
        pre_correct_mod_div_s = torch.where(z < self.nullify_threshold, self.elem_zeros + 1, self.elem_zeros)
        pre_correct_mod_div_s = pmod(pre_correct_mod_div_s - self.pre_mod_div_c, self.q_23)
        self.common.pre_corr_mod_s.send(pre_correct_mod_div_s)

    def offline_recv(self):
        for blob in self.common.offline_server_send:
            blob.prepare_recv()

    def online_recv(self):
        for blob in self.common.online_server_send:
            blob.prepare_recv()

    def offline(self):
        self.offline_recv()

        self.beta_i_c = gen_unirand_int_grain(0, self.q_16 - 1, self.decomp_bit_shape).to(Config.device)
        self.delta_b_c = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).to(Config.device)
        self.z_work_c = gen_unirand_int_grain(0, self.q_23 - 1, self.num_elem).to(Config.device)
        self.beta_i_zeros = torch.zeros_like(self.beta_i_c)
        self.fast_ones = torch.ones(self.num_elem).to(Config.device)
        self.fast_zeros = torch.zeros(self.num_elem).to(Config.device)
        self.fast_ones_c_i = torch.ones(self.sum_shape).float().to(Config.device)
        self.fast_zeros_c_i = torch.zeros(self.sum_shape).float().to(Config.device)

        self.common.beta_i_c.send_from_torch(self.beta_i_c)
        self.common.delta_b_c.send_from_torch(self.delta_b_c)
        self.common.z_work_c.send_from_torch(self.z_work_c)

        if self.is_shuffle:
            self.sum_c_refresher = EncRefresherClient(self.sum_shape, self.fhe_builder_16, self.sub_name("shuffle_refresher"))

        self.mod_div_offline()

        refresher_ab_xor_c = EncRefresherClient(
            self.decomp_bit_shape, self.fhe_builder_16, self.common.sub_name("refresher_ab_xor_c"))
        refresher_ab_xor_c.response()
        self.sum_c_i_offline()

        self.c_i_c = self.common.c_i_c.get_recv_decrypt()
        self.delta_xor_c = self.common.delta_xor_c.get_recv_decrypt()
        self.dgk_x_leq_y_c = self.common.dgk_x_leq_y_c.get_recv_decrypt()
        self.dgk_x_leq_y_c = pmod(self.dgk_x_leq_y_c + self.correct_mod_div_work_c, self.q_23)

        self.online_recv()
        torch_sync()

    def online(self, y_sub_x_c):
        z_s = self.common.z_s.get_recv()
        z = pmod(z_s + y_sub_x_c.to(Config.device), self.q_23)
        self.z = z
        beta = pmod(z, self.work_range)
        beta_i = self.common.decomp_to_bit(beta, res=self.beta_i_zeros).to(Config.device)
        beta_i_s = pmod(beta_i.to(Config.device) - self.beta_i_c.to(Config.device), self.q_16)
        self.common.beta_i_s.send(beta_i_s)

        c_i_s = self.common.c_i_s.get_recv()
        c_i = pmod(c_i_s + self.c_i_c, self.q_16)
        check_zeros = torch.where(c_i == 0, torch.tensor(1).to(Config.device), torch.tensor(0).to(Config.device))
        delta_b = torch.where(torch.sum(check_zeros, 0) > 0,
                              self.fast_ones, self.fast_zeros)
        delta_b_s = pmod(delta_b - self.delta_b_c, self.q_23)
        self.common.delta_b_s.send(delta_b_s)
        z_work_s = pmod(z // self.work_range - self.z_work_c, self.q_23)
        self.common.z_work_s.send(z_work_s)
        self.mod_div_online(z)


def test_dgk(input_sid, master_address, master_port, num_elem=2**17):
    print("\nTest for Dgk Basic: Start")
    data_bit = 20
    work_bit = 20
    # q_23 = 786433
    # q_23 = 8273921
    # q_23 = 4079617
    # n_23, q_23 = 8192, 7340033
    n_23, q_23 = Config.n_23, Config.q_23
    # n_16, q_16 = 2048, 12289
    # n_16, q_16 = 8192, 65537
    n_16, q_16 = Config.n_16, Config.q_16
    print(f"Number of element: {num_elem}")

    data_range = 2 ** data_bit
    work_range = 2 ** work_bit

    def check_correctness(x, y, dgk_x_leq_y_s, dgk_x_leq_y_c):
        x = torch.where(x < q_23 // 2, x, x - q_23).to(Config.device)
        y = torch.where(y < q_23 // 2, y, y - q_23).to(Config.device)
        expected_x_leq_y = (x <= y)
        dgk_x_leq_y_recon = pmod(dgk_x_leq_y_s + dgk_x_leq_y_c, q_23)
        compare_expected_actual(expected_x_leq_y, dgk_x_leq_y_recon, name="DGK x <= y", get_relative=True)
        print(torch.sum(expected_x_leq_y != dgk_x_leq_y_recon))

    def check_correctness_mod_div(r, z, correct_mod_div_work_s, correct_mod_div_work_c):
        elem_zeros = torch.zeros(num_elem).to(Config.device)
        expected = torch.where(r > z, q_23//work_range + elem_zeros, elem_zeros)
        actual = pmod(correct_mod_div_work_s + correct_mod_div_work_c, q_23)
        compare_expected_actual(expected, actual, get_relative=True, name="mod_div_online")

    def test_server():
        rank = Config.server_rank
        init_communicate(Config.server_rank, master_address=master_address, master_port=master_port)
        warming_up_cuda()
        traffic_record = TrafficRecord()

        fhe_builder_16 = FheBuilder(q_16, Config.n_16)
        fhe_builder_23 = FheBuilder(q_23, Config.n_23)
        comm_fhe_16 = CommFheBuilder(rank, fhe_builder_16, "fhe_builder_16")
        comm_fhe_23 = CommFheBuilder(rank, fhe_builder_23, "fhe_builder_23")
        torch_sync()
        comm_fhe_16.recv_public_key()
        comm_fhe_23.recv_public_key()
        comm_fhe_16.wait_and_build_public_key()
        comm_fhe_23.wait_and_build_public_key()

        dgk = DgkBitServer(num_elem, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, "DgkBitTest")

        x_blob = BlobTorch(num_elem, torch.float, dgk.comm_base, "x")
        y_blob = BlobTorch(num_elem, torch.float, dgk.comm_base, "y")
        y_sub_x_s_blob = BlobTorch(num_elem, torch.float, dgk.comm_base, "y_sub_x_s")
        x_blob.prepare_recv()
        y_blob.prepare_recv()
        y_sub_x_s_blob.prepare_recv()
        torch_sync()
        x = x_blob.get_recv()
        y = y_blob.get_recv()
        y_sub_x_s = y_sub_x_s_blob.get_recv()

        torch_sync()
        with NamedTimerInstance("Server Offline"):
            dgk.offline()
        # y_sub_x_s = pmod(y_s.to(Config.device) - x_s.to(Config.device), q_23)
        torch_sync()
        traffic_record.reset("server-offline")

        with NamedTimerInstance("Server Online"):
            dgk.online(y_sub_x_s)
        traffic_record.reset("server-online")

        dgk_x_leq_y_c_blob = BlobTorch(num_elem, torch.float, dgk.comm_base, "dgk_x_leq_y_c")
        correct_mod_div_work_c_blob = BlobTorch(num_elem, torch.float, dgk.comm_base, "correct_mod_div_work_c")
        z_blob = BlobTorch(num_elem, torch.float, dgk.comm_base, "z")
        dgk_x_leq_y_c_blob.prepare_recv()
        correct_mod_div_work_c_blob.prepare_recv()
        z_blob.prepare_recv()
        torch_sync()
        dgk_x_leq_y_c = dgk_x_leq_y_c_blob.get_recv()
        correct_mod_div_work_c = correct_mod_div_work_c_blob.get_recv()
        z = z_blob.get_recv()
        check_correctness(x, y, dgk.dgk_x_leq_y_s, dgk_x_leq_y_c)
        check_correctness_mod_div(dgk.r, z, dgk.correct_mod_div_work_s, correct_mod_div_work_c)
        end_communicate()

    def test_client():
        rank = Config.client_rank
        init_communicate(rank, master_address=master_address, master_port=master_port)
        warming_up_cuda()
        traffic_record = TrafficRecord()

        fhe_builder_16 = FheBuilder(q_16, n_16)
        fhe_builder_23 = FheBuilder(q_23, n_23)
        fhe_builder_16.generate_keys()
        fhe_builder_23.generate_keys()
        comm_fhe_16 = CommFheBuilder(rank, fhe_builder_16, "fhe_builder_16")
        comm_fhe_23 = CommFheBuilder(rank, fhe_builder_23, "fhe_builder_23")
        torch_sync()
        comm_fhe_16.send_public_key()
        comm_fhe_23.send_public_key()

        dgk = DgkBitClient(num_elem, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, "DgkBitTest")

        x = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem)
        y = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem)
        x_c = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem)
        y_c = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem)
        x_s = pmod(x - x_c, q_23)
        y_s = pmod(y - y_c, q_23)
        y_sub_x_s = pmod(y_s - x_s, q_23)

        x_blob = BlobTorch(num_elem, torch.float, dgk.comm_base, "x")
        y_blob = BlobTorch(num_elem, torch.float, dgk.comm_base, "y")
        y_sub_x_s_blob = BlobTorch(num_elem, torch.float, dgk.comm_base, "y_sub_x_s")
        torch_sync()
        x_blob.send(x)
        y_blob.send(y)
        y_sub_x_s_blob.send(y_sub_x_s)

        torch_sync()
        with NamedTimerInstance("Client Offline"):
            dgk.offline()
        y_sub_x_c = pmod(y_c - x_c, q_23)
        traffic_record.reset("client-offline")
        torch_sync()

        with NamedTimerInstance("Client Online"):
            dgk.online(y_sub_x_c)
        traffic_record.reset("client-online")

        dgk_x_leq_y_c_blob = BlobTorch(num_elem, torch.float, dgk.comm_base, "dgk_x_leq_y_c")
        correct_mod_div_work_c_blob = BlobTorch(num_elem, torch.float, dgk.comm_base, "correct_mod_div_work_c")
        z_blob = BlobTorch(num_elem, torch.float, dgk.comm_base, "z")
        torch_sync()
        dgk_x_leq_y_c_blob.send(dgk.dgk_x_leq_y_c)
        correct_mod_div_work_c_blob.send(dgk.correct_mod_div_work_c)
        z_blob.send(dgk.z)
        end_communicate()

    if input_sid == Config.both_rank:
        marshal_funcs([test_server, test_client])
    elif input_sid == Config.server_rank:
        marshal_funcs([test_server])
    elif input_sid == Config.client_rank:
        marshal_funcs([test_client])

    print("\nTest for Dgk Basic: End")

if __name__ == "__main__":
    input_sid, master_address, master_port, test_to_run = argparser_distributed()
    sys.stdout = Logger()

    num_repeat = 10
    num_elem_try = [10000, 40000] + [2 ** i for i in range(10, 19)]
    # num_elem_try = [10000, 40000] + [2 ** i for i in range(10, 19)]
    # num_repeat = 1
    # num_elem_try = [2 ** 14]

    for _, num_elem in product(range(num_repeat), num_elem_try):
        test_dgk(input_sid, master_address, master_port, num_elem=num_elem)
