import sys
from itertools import product
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from seal import uIntVector, Plaintext, Ciphertext

from comm import BlobFheRawCts, CommBase, CommFheBuilder, PhaseProtocolServer, PhaseProtocolClient, init_communicate, \
    torch_sync, BlobTorch, end_communicate, PhaseProtocolCommon
from config import Config
from fhe import FheBuilder, encrypt_zeros
from logger_utils import Logger
from ntt import get_pow_2_ceil, NttMatmul, pad_to_torch
from numbertheoretictransform import find_modulus, find_primitive_root
from timer_utils import NamedTimerInstance
from torch_utils import gen_unirand_int_grain, compare_expected_actual, pmod, get_prod, warming_up_cuda, marshal_funcs, \
    generate_random_mask, argparser_distributed


def calc_output_hw(img_hw, filter_hw, padding):
    return img_hw + 2 * padding - (filter_hw - 1)


class Conv2dNttParamBase(object):
    def __init__(self, modulus, data_range, img_hw, filter_hw, num_input_channel, num_output_channel, name, padding):
        self.modulus = modulus
        self.data_range = data_range
        self.img_hw = img_hw
        self.filter_hw = filter_hw
        self.padding = padding
        self.num_input_channel = num_input_channel
        self.num_output_channel = num_output_channel
        self.name = name
        self.degree = Config.n_23
        # self.padded_hw = get_pow_2_ceil(self.img_hw + self.filter_hw)
        self.padded_hw = self.find_min_len(self.img_hw + self.filter_hw - 1, self.modulus)
        self.num_elem_in_padded = self.padded_hw ** 2
        self.num_elem_in_image = img_hw ** 2
        self.conv_hw = img_hw + 2 * padding
        self.output_hw = calc_output_hw(img_hw, filter_hw, padding)
        self.output_offset = filter_hw - 2
        self.y_shape = [self.num_output_channel, self.output_hw, self.output_hw]

        self.num_rotation = self.degree // self.num_elem_in_padded
        self.num_input_batch = int(ceil(self.num_input_channel / self.num_rotation)) * self.num_rotation
        self.num_output_batch = int(ceil(self.num_output_channel / self.num_rotation))
        self.input_shape = torch.Size([self.num_input_channel, self.img_hw, self.img_hw])
        self.weight_shape = torch.Size([self.num_output_channel, self.num_input_channel, self.filter_hw, self.filter_hw])
        self.output_shape = torch.Size([self.num_output_channel, self.output_hw, self.output_hw])

    def find_min_len(self, len_vec, modulus):
        start = len_vec
        while modulus != find_modulus(start, modulus):
            start += 1
        return start

    def sub_name(self, sub_name: str) -> str:
        return self.name + '_' + self.class_name + '_' + sub_name


# Reference: https://pytorch.org/docs/stable/nn.html#conv2d
class Conv2dNtt(Conv2dNttParamBase):
    ntted_x = None
    ntted_w = None
    def __init__(self, modulus, data_range, img_hw, filter_hw, num_input_channel, num_output_channel, name, padding=1):
        super().__init__(modulus, data_range, img_hw, filter_hw, num_input_channel, num_output_channel, name, padding)

        self.ntt_matmul = NttMatmul(self.modulus, self.padded_hw, self.data_range)
        self.padded = torch.zeros([self.padded_hw, self.padded_hw]).double()

    def refresh_padded(self):
        self.padded = torch.zeros([self.padded_hw, self.padded_hw]).double()

    def pad_to_torch(self, x):
        img_hw = len(x)
        self.padded[:img_hw, :img_hw] = x
        return self.padded

    def transform_x_single_channel(self, x):
        return self.ntt_matmul.ntt2d(pad_to_torch(x, self.padded_hw))

    def load_and_ntt_x(self, x, is_batched=False):
        assert(len(x.shape) == 3)
        if is_batched:
            padded = torch.zeros([self.num_input_channel, self.padded_hw, self.padded_hw], device="cuda", dtype=torch.float)
            padded[:, :self.img_hw, :self.img_hw] = x.cuda()
            padded = padded.double()
            ntt_mat = self.ntt_matmul.ntt_mat.cuda().double()
            ntted = torch.matmul(padded, ntt_mat.t()).fmod_(self.modulus)
            ntted = torch.matmul(ntt_mat, ntted).fmod_(self.modulus)
            self.ntted_x = ntted.float().cpu()
        else:
            x = x.double()
            self.ntted_x = torch.zeros([self.num_input_channel, self.padded_hw, self.padded_hw])
            self.refresh_padded()
            for i in range(self.num_input_channel):
                self.ntted_x[i, :, :] = self.transform_x_single_channel(x[i])
            self.refresh_padded()

    def index_output_crop(self):
        return self.output_offset, self.output_hw + self.output_offset


    def ntt_output_masking(self, masking):
        masking = masking.double()
        ntted = torch.zeros([self.num_output_channel, self.padded_hw, self.padded_hw])
        padded = torch.zeros([self.padded_hw, self.padded_hw]).double()
        for i in range(self.num_output_channel):
            start_hw, end_hw = self.index_output_crop()
            padded[start_hw:end_hw, start_hw:end_hw] = masking[i]
            ntted[i, :, :] = self.ntt_matmul.ntt2d(padded)
        return ntted

    def transform_w_single_channel(self, w):
        return self.ntt_matmul.ntt2d(self.pad_to_torch(w.rot90(2)))
        # return self.ntt_matmul.ntt2d(pad_to_torch(w.rot90(2), self.padded_hw))

    def load_and_ntt_w(self, w, is_batched=True):
        assert(len(w.shape) == 4)
        if is_batched:
            padded = torch.zeros([self.num_output_channel, self.num_input_channel, self.padded_hw, self.padded_hw], device="cuda", dtype=torch.double)
            padded[:, :, :self.filter_hw, :self.filter_hw] = w.cuda().rot90(2, [2, 3]).double()
            ntt_mat = self.ntt_matmul.ntt_mat.cuda().double()
            ntted_w = torch.matmul(padded, ntt_mat.t()).fmod_(self.modulus)
            ntted_w = torch.matmul(ntt_mat, ntted_w).fmod_(self.modulus)
            self.ntted_w = ntted_w.float().cpu()
        else:
            w = w.cuda().double()
            self.refresh_padded()
            self.ntted_w = torch.zeros([self.num_output_channel, self.num_input_channel, self.padded_hw, self.padded_hw])
            for i, j in product(range(self.num_output_channel), range(self.num_input_channel)):
                self.ntted_w[i, j, :, :] = self.transform_w_single_channel(w[i, j])
            self.refresh_padded()

    def conv2d_ntted_single_channel(self, ntted_x, ntted_w):
        return pmod(ntted_x.double() * ntted_w.double(), self.ntt_matmul.mod)

    def transform_y_single_channel(self, y):
        reved = self.ntt_matmul.intt2d(y.double())
        start_hw, end_hw = self.index_output_crop()
        return reved[start_hw:end_hw, start_hw:end_hw]

    def conv2d_loaded(self):
        y = torch.zeros([self.num_output_channel, self.output_hw, self.output_hw]).double()
        for i, j in product(range(self.num_output_channel), range(self.num_input_channel)):
            single_y = self.conv2d_ntted_single_channel(self.ntted_x[j], self.ntted_w[i, j])
            y[i, :, :] += self.transform_y_single_channel(single_y)
        return pmod(y, self.modulus)

    def conv2d(self, x, w):
        self.load_and_ntt_w(w)
        self.load_and_ntt_x(x)

        sub_x = x[0]
        sub_ntted_x = self.ntted_x[0]
        inv_sub_ntted_x = self.ntt_matmul.intt2d(sub_ntted_x.double())
        trun_inv_sub_ntted_x = inv_sub_ntted_x[:self.img_hw, :self.img_hw]
        compare_expected_actual(pmod(sub_x, self.modulus), pmod(trun_inv_sub_ntted_x, self.modulus), get_relative=True, name="sub_x")

        sub_w = w[0, 0]
        sub_ntted_w = self.ntted_w[0, 0]
        inv_sub_ntted_w = self.ntt_matmul.intt2d(sub_ntted_w.double())
        trun_inv_sub_ntted_w = inv_sub_ntted_w[:self.filter_hw, :self.filter_hw]
        expected = pmod(sub_w, self.modulus).rot90(2)
        actual = pmod(trun_inv_sub_ntted_w, self.modulus)
        compare_expected_actual(expected, actual, get_relative=True, name="sub_w")

        dotted = self.conv2d_ntted_single_channel(sub_ntted_x, sub_ntted_w)
        sub_y = self.transform_y_single_channel(dotted)

        # sub_w = w[0, 0]
        # sub_ntted_w = self.ntted_w[0]
        return self.conv2d_loaded()


def test_ntt_conv():
    modulus = 786433
    img_hw = 16
    filter_hw = 3
    padding = 1
    num_input_channel = 64
    num_output_channel = 128
    data_bit = 17
    data_range = 2 ** data_bit

    x_shape = [num_input_channel, img_hw, img_hw]
    w_shape = [num_output_channel, num_input_channel, filter_hw, filter_hw]

    x = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(x_shape)).reshape(x_shape)
    w = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(w_shape)).reshape(w_shape)
    # x = torch.arange(get_prod(x_shape)).reshape(x_shape)
    # w = torch.arange(get_prod(w_shape)).reshape(w_shape)

    conv2d_ntt = Conv2dNtt(modulus, data_range, img_hw, filter_hw, num_input_channel, num_output_channel, "test_conv2d_ntt", padding)
    y = conv2d_ntt.conv2d(x, w)

    with NamedTimerInstance("ntt x"):
        conv2d_ntt.load_and_ntt_x(x)
    with NamedTimerInstance("ntt w"):
        conv2d_ntt.load_and_ntt_w(w)
    with NamedTimerInstance("conv2d"):
        y = conv2d_ntt.conv2d_loaded()
    actual = pmod(y, modulus)
    # print("actual\n", actual)

    torch_x = x.reshape([1] + x_shape).double()
    torch_w = w.reshape(w_shape).double()
    with NamedTimerInstance("Conv2d Torch"):
        expected = F.conv2d(torch_x, torch_w, padding=padding)
        expected = pmod(expected.reshape(conv2d_ntt.y_shape), modulus)
    # print("expected", expected)
    compare_expected_actual(expected, actual, name="ntt", get_relative=True, show_where_err=False)


def test_rotation():
    modulus = Config.q_23
    degree = Config.n_23
    fhe_builder = FheBuilder(modulus, degree)
    fhe_builder.generate_keys()
    fhe_builder.generate_galois_keys()

    # x = gen_unirand_int_grain(0, modulus-1, degree)
    x = torch.arange(degree)
    with NamedTimerInstance("Fhe Encrypt"):
        enc = fhe_builder.build_enc_from_torch(x)
    enc_less = fhe_builder.build_enc_from_torch(x)
    plain = fhe_builder.build_plain_from_torch(x)

    fhe_builder.noise_budget(enc, "before mul")
    with NamedTimerInstance("ep mult"):
        enc *= plain
        enc_less *= plain
    fhe_builder.noise_budget(enc, "after mul")
    with NamedTimerInstance("ee add"):
        for i in range(128):
            enc += enc_less
    fhe_builder.noise_budget(enc, "after add")
    with NamedTimerInstance("rot"):
        fhe_builder.evaluator.rotate_rows_inplace(enc.cts[0], 64, fhe_builder.galois_keys)
    fhe_builder.noise_budget(enc, "after rot")
    print(fhe_builder.decrypt_to_torch(enc))


# batch = piece || piece || piece || ...
# input or output = batch || batch || batch || ...
class Conv2dFheNttSingleThread(Conv2dNttParamBase):
    def __init__(self, modulus, data_range, img_hw, filter_hw, num_input_channel, num_output_channel,
                 fhe_builder: FheBuilder, name, padding):
        super().__init__(modulus, data_range, img_hw, filter_hw, num_input_channel, num_output_channel, name, padding)
        self.fhe_builder = fhe_builder
        self.input_cts = None
        self.output_mask_s = None
        self.batch_encoder = self.fhe_builder.batch_encoder
        self.encryptor = self.fhe_builder.encryptor
        self.evaluator = self.fhe_builder.evaluator
        self.decryptor = self.fhe_builder.decryptor

        # assert(self.fhe_builder.degree == 8192)
        assert(self.num_elem_in_padded <= self.fhe_builder.degree)

        self.conv2d_ntt = Conv2dNtt(modulus, data_range, img_hw, filter_hw, num_input_channel, num_output_channel,
                                    name, padding=padding)

    def index_input_piece_to_channel(self, index_input_batch, index_piece):
        num_total_rot = self.num_rotation
        base_channel = (index_input_batch // num_total_rot) * num_total_rot
        addition_channel = (index_input_batch % num_total_rot + index_piece) % num_total_rot
        channel = base_channel + addition_channel
        if channel >= self.num_input_channel:
            return False
        return channel

    def index_output_piece_to_channel(self, index_output_batch, index_piece):
        num_total_rot = self.num_rotation
        base_channel = index_output_batch * num_total_rot
        addition_channel = index_piece
        channel = base_channel + addition_channel
        if channel >= self.num_output_channel:
            return False
        return channel

    def index_weight_piece_to_channel(self, index_output_batch, index_input_batch, index_piece):
        input_channel = self.index_input_piece_to_channel(index_input_batch, index_piece)
        output_channel = self.index_output_piece_to_channel(index_output_batch, index_piece)
        return input_channel, output_channel

    def encode_input_to_fhe_batch(self, input_tensor):
        assert(input_tensor.shape == self.input_shape)
        self.conv2d_ntt.load_and_ntt_x(input_tensor)
        ntted_input = self.conv2d_ntt.ntted_x
        self.input_cts = [Ciphertext() for _ in range(self.num_input_batch)]
        pod_vector = uIntVector()
        pt = Plaintext()
        for index_batch in range(self.num_input_batch):
            encoding_tensor = torch.zeros(self.degree, dtype=torch.float)
            for index_piece in range(self.num_rotation):
                index_input_channel = self.index_input_piece_to_channel(index_batch, index_piece)
                if index_input_channel is False:
                    continue
                span = self.num_elem_in_padded
                start_piece = index_piece * span
                encoding_tensor[start_piece: start_piece+span] = ntted_input[index_input_channel].reshape(-1)
            encoding_tensor = pmod(encoding_tensor, self.modulus)
            pod_vector.from_np(encoding_tensor.numpy().astype(np.uint64))
            self.batch_encoder.encode(pod_vector, pt)
            self.encryptor.encrypt(pt, self.input_cts[index_batch])

    def compute_conv2d(self, weight_tensor):
        assert(weight_tensor.shape == self.weight_shape)
        self.conv2d_ntt.load_and_ntt_w(weight_tensor)
        ntted_weight = self.conv2d_ntt.ntted_w
        pod_vector = uIntVector()
        pt_w = Plaintext()
        self.output_cts = encrypt_zeros(self.num_output_batch, self.batch_encoder, self.encryptor, self.degree)
        for idx_output_batch, idx_input_batch in product(range(self.num_output_batch), range(self.num_input_batch)):
            encoding_tensor = torch.zeros(self.degree, dtype=torch.float)
            is_w_changed = False
            for index_piece in range(self.num_rotation):
                span = self.num_elem_in_padded
                start_piece = index_piece * span
                index_input_channel, index_output_channel = \
                    self.index_weight_piece_to_channel(idx_output_batch, idx_input_batch, index_piece)
                if index_input_channel is False or index_output_channel is False:
                    continue
                is_w_changed = True
                encoding_tensor[start_piece: start_piece+span] = \
                    ntted_weight[index_output_channel, index_input_channel].reshape(-1)
            if not is_w_changed:
                continue
            encoding_tensor = pmod(encoding_tensor, self.modulus)
            pod_vector.from_np(encoding_tensor.numpy().astype(np.uint64))
            self.batch_encoder.encode(pod_vector, pt_w)
            sub_dotted = Ciphertext(self.input_cts[idx_input_batch])
            # print(idx_output_batch, idx_input_batch)
            # print("noise", self.decryptor.invariant_noise_budget(self.input_cts[idx_input_batch]))
            # print("noise", self.decryptor.invariant_noise_budget(sub_dotted))
            self.evaluator.multiply_plain_inplace(sub_dotted, pt_w)
            self.evaluator.add_inplace(self.output_cts[idx_output_batch], sub_dotted)
            del sub_dotted

    def masking_output(self):
        self.output_mask_s = gen_unirand_int_grain(0, self.modulus - 1, get_prod(self.y_shape)).reshape(self.y_shape)
        # self.output_mask_s = torch.ones(self.y_shape)
        ntted_mask = self.conv2d_ntt.ntt_output_masking(self.output_mask_s)

        pod_vector = uIntVector()
        pt = Plaintext()
        for idx_output_batch in range(self.num_output_batch):
            encoding_tensor = torch.zeros(self.degree, dtype=torch.float)
            for index_piece in range(self.num_rotation):
                span = self.num_elem_in_padded
                start_piece = index_piece * span
                index_output_channel = self.index_output_piece_to_channel(idx_output_batch, index_piece)
                if index_output_channel is False:
                    continue
                encoding_tensor[start_piece: start_piece+span] = ntted_mask[index_output_channel].reshape(-1)
            encoding_tensor = pmod(encoding_tensor, self.modulus)
            pod_vector.from_np(encoding_tensor.numpy().astype(np.uint64))
            self.batch_encoder.encode(pod_vector, pt)
            self.evaluator.add_plain_inplace(self.output_cts[idx_output_batch], pt)

    def decode_output_from_fhe_batch(self):
        output_tensor = torch.zeros(self.output_shape)
        pod_vector = uIntVector()
        pt = Plaintext()
        cts = self.output_cts
        for index_output_batch in range(self.num_output_batch):
            self.decryptor.decrypt(cts[index_output_batch], pt)
            self.batch_encoder.decode(pt, pod_vector)
            arr = np.array(pod_vector, copy=False)
            arr = torch.from_numpy(arr.astype(np.float))
            for index_rot in range(self.num_rotation):
                index_output_channel = self.index_output_piece_to_channel(index_output_batch, index_rot)
                if index_output_channel is False:
                    continue
                span = self.num_elem_in_padded
                start_piece = index_rot * span
                sub_ntted_y = arr[start_piece: start_piece+span].reshape([self.padded_hw, self.padded_hw])
                sub_y = self.conv2d_ntt.transform_y_single_channel(sub_ntted_y)
                output_tensor[index_output_channel].copy_(sub_y)

        return output_tensor


def test_conv2d_fhe_ntt_single_thread():
    modulus = 786433
    img_hw = 16
    filter_hw = 3
    padding = 1
    num_input_channel = 64
    num_output_channel = 128
    data_bit = 17
    data_range = 2 ** data_bit

    x_shape = [num_input_channel, img_hw, img_hw]
    w_shape = [num_output_channel, num_input_channel, filter_hw, filter_hw]

    fhe_builder = FheBuilder(modulus, Config.n_23)
    fhe_builder.generate_keys()

    # x = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(x_shape)).reshape(x_shape)
    w = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(w_shape)).reshape(w_shape)
    x = gen_unirand_int_grain(0, modulus, get_prod(x_shape)).reshape(x_shape)
    # x = torch.arange(get_prod(x_shape)).reshape(x_shape)
    # w = torch.arange(get_prod(w_shape)).reshape(w_shape)

    warming_up_cuda()
    prot = Conv2dFheNttSingleThread(modulus, data_range, img_hw, filter_hw, num_input_channel, num_output_channel,
                                    fhe_builder, "test_conv2d_fhe_ntt", padding)

    print("prot.num_input_batch", prot.num_input_batch)
    print("prot.num_output_batch", prot.num_output_batch)

    with NamedTimerInstance("encoding x"):
        prot.encode_input_to_fhe_batch(x)
    with NamedTimerInstance("conv2d with w"):
        prot.compute_conv2d(w)
    with NamedTimerInstance("conv2d masking output"):
        prot.masking_output()
    with NamedTimerInstance("decoding output"):
        y = prot.decode_output_from_fhe_batch()
    # actual = pmod(y, modulus)
    actual = pmod(y - prot.output_mask_s, modulus)
    # print("actual\n", actual)


    torch_x = x.reshape([1] + x_shape).double()
    torch_w = w.reshape(w_shape).double()
    with NamedTimerInstance("Conv2d Torch"):
        expected = F.conv2d(torch_x, torch_w, padding=padding)
        expected = pmod(expected.reshape(prot.output_shape), modulus)
    # print("expected", expected)
    compare_expected_actual(expected, actual, name="test_conv2d_fhe_ntt_single_thread", get_relative=True)


class Conv2dFheNttBase(Conv2dNttParamBase):
    class_name = "Conv2dFheNtt"
    def __init__(self, modulus, fhe_builder: FheBuilder, data_range, img_hw, filter_hw,
                 num_input_channel, num_output_channel, name: str, rank, padding):
        super(Conv2dFheNttBase, self).__init__(modulus, data_range, img_hw, filter_hw,
                                               num_input_channel, num_output_channel, name, padding)
        self.fhe_builder = fhe_builder
        self.rank = rank
        self.comm_base = CommBase(rank, self.sub_name("comm_base"))
        self.comm_fhe = CommFheBuilder(rank, fhe_builder, self.sub_name("comm_fhe"))
        self.compute_core = Conv2dFheNttSingleThread(modulus, data_range, img_hw, filter_hw,
                                                     num_input_channel, num_output_channel, fhe_builder, name, padding)

        assert(self.modulus == self.fhe_builder.modulus)

        self.blob_input_cts = BlobFheRawCts(self.num_input_batch, self.comm_fhe, self.sub_name("input_cts"))
        self.blob_output_cts = BlobFheRawCts(self.num_output_batch, self.comm_fhe, self.sub_name("output_cts"))

        self.offline_server_send = [self.blob_output_cts]
        self.offline_client_send = [self.blob_input_cts]
        self.online_server_send = []
        self.online_client_send = []


class Conv2dFheNttServer(Conv2dFheNttBase, PhaseProtocolServer):
    def __init__(self, modulus, fhe_builder: FheBuilder, data_range, img_hw, filter_hw,
                 num_input_channel, num_output_channel, name: str, padding=1):
        super().__init__(modulus, fhe_builder, data_range, img_hw, filter_hw,
                         num_input_channel, num_output_channel, name, Config.server_rank, padding)

    def offline(self, weight_tensor):
        PhaseProtocolServer.offline(self)
        self.compute_core.input_cts = self.blob_input_cts.get_recv()
        self.compute_core.compute_conv2d(weight_tensor)
        self.compute_core.masking_output()
        self.blob_output_cts.send(self.compute_core.output_cts)
        self.output_mask_s = self.compute_core.output_mask_s.cuda().double()


class Conv2dFheNttClient(Conv2dFheNttBase, PhaseProtocolClient):
    def __init__(self, modulus, fhe_builder: FheBuilder, data_range, img_hw, filter_hw,
                 num_input_channel, num_output_channel, name: str, padding=1):
        super().__init__(modulus, fhe_builder, data_range, img_hw, filter_hw,
                         num_input_channel, num_output_channel, name, Config.client_rank, padding)

    def offline(self, input_mask):
        PhaseProtocolServer.offline(self)
        self.compute_core.encode_input_to_fhe_batch(input_mask)
        self.blob_input_cts.send(self.compute_core.input_cts)
        self.compute_core.output_cts = self.blob_output_cts.get_recv()
        self.output_c = self.compute_core.decode_output_from_fhe_batch().cuda().double()


def test_conv2d_fhe_ntt_comm():
    test_name = "Conv2d Fhe NTT Comm"
    print(f"\nTest for {test_name}: Start")
    modulus = 786433
    img_hw = 2
    filter_hw = 3
    padding = 1
    num_input_channel = 512
    num_output_channel = 512
    data_bit = 17
    data_range = 2 ** data_bit
    print(f"Setting: img_hw {img_hw}, "
          f"filter_hw: {filter_hw}, "
          f"num_input_channel: {num_input_channel}, "
          f"num_output_channel: {num_output_channel}")

    x_shape = [num_input_channel, img_hw, img_hw]
    w_shape = [num_output_channel, num_input_channel, filter_hw, filter_hw]

    fhe_builder = FheBuilder(modulus, Config.n_23)
    fhe_builder.generate_keys()

    weight = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(w_shape)).reshape(w_shape)
    input_mask = gen_unirand_int_grain(0, modulus - 1, get_prod(x_shape)).reshape(x_shape)
    # input_mask = torch.arange(get_prod(x_shape)).reshape(x_shape)

    def check_correctness_offline(x, w, output_mask, output_c):
        actual = pmod(output_c.cuda() - output_mask.cuda(), modulus)
        torch_x = x.reshape([1] + x_shape).cuda().double()
        torch_w = w.reshape(w_shape).cuda().double()
        expected = F.conv2d(torch_x, torch_w, padding=padding)
        expected = pmod(expected.reshape(output_mask.shape), modulus)
        compare_expected_actual(expected, actual, name=test_name + " offline", get_relative=True)

    def test_server():
        init_communicate(Config.server_rank)
        prot = Conv2dFheNttServer(modulus, fhe_builder, data_range, img_hw, filter_hw,
                                  num_input_channel, num_output_channel, "test_conv2d_fhe_ntt_comm", padding=padding)
        with NamedTimerInstance("Server Offline"):
            prot.offline(weight)
            torch_sync()

        blob_output_c = BlobTorch(prot.output_shape, torch.float, prot.comm_base, "output_c")
        blob_output_c.prepare_recv()
        torch_sync()
        output_c = blob_output_c.get_recv()
        check_correctness_offline(input_mask, weight, prot.output_mask_s, output_c)

        end_communicate()

    def test_client():
        init_communicate(Config.client_rank)
        prot = Conv2dFheNttClient(modulus, fhe_builder, data_range, img_hw, filter_hw,
                                  num_input_channel, num_output_channel, "test_conv2d_fhe_ntt_comm", padding=padding)
        with NamedTimerInstance("Client Offline"):
            prot.offline(input_mask)
            torch_sync()

        blob_output_c = BlobTorch(prot.output_shape, torch.float, prot.comm_base, "output_c")
        torch_sync()
        blob_output_c.send(prot.output_c)
        end_communicate()

    marshal_funcs([test_server, test_client])
    print(f"\nTest for {test_name}: End")


class Conv2dSecureCommon(Conv2dNttParamBase, PhaseProtocolCommon):
    class_name = "Conv2dSecure"
    def __init__(self, modulus, fhe_builder: FheBuilder, data_range, img_hw, filter_hw,
                 num_input_channel, num_output_channel, name: str, rank, padding):
        super().__init__(modulus, data_range, img_hw, filter_hw, num_input_channel, num_output_channel, name, padding)
        self.fhe_builder = fhe_builder
        self.rank = rank
        self.comm_base = CommBase(rank, self.sub_name("comm_base"))
        self.comm_fhe = CommFheBuilder(rank, fhe_builder, self.sub_name("comm_fhe"))

        self.offline_server_send = []
        self.offline_client_send = []
        self.online_server_send = []
        self.online_client_send = []


class Conv2dSecureServer(Conv2dSecureCommon):
    weight = None
    bias = None
    output_mask_s = None
    output_s = None
    torch_w = None
    def __init__(self, modulus, fhe_builder: FheBuilder, data_range, img_hw, filter_hw,
                 num_input_channel, num_output_channel, name: str, padding=1):
        super().__init__(modulus, fhe_builder, data_range, img_hw, filter_hw,
                         num_input_channel, num_output_channel, name, Config.server_rank, padding)

        self.fhe_ntt = Conv2dFheNttServer(modulus, fhe_builder, data_range, img_hw, filter_hw,
                                          num_input_channel, num_output_channel, name, padding=padding)

    def offline(self, weight, bias=None):
        self.weight = weight
        self.bias = bias.cuda().double() if bias is not None else None
        self.fhe_ntt.offline(weight)
        self.output_mask_s = self.fhe_ntt.output_mask_s.cuda().double()
        self.torch_w = weight.reshape(self.weight_shape).cuda().double()

        # warming up
        warm_up_x = generate_random_mask(self.modulus, [1] + list(self.input_shape)).cuda().double()
        warm_up_y = F.conv2d(warm_up_x, self.torch_w, padding=self.padding)

    def online(self, input_s):
        input_s = input_s.reshape([1] + list(self.input_shape)).cuda().double()
        y_s = F.conv2d(input_s, self.torch_w, padding=self.padding, bias=self.bias)
        y_s = pmod(y_s.reshape(self.y_shape) - self.output_mask_s , self.modulus)
        self.output_s = y_s


class Conv2dSecureClient(Conv2dSecureCommon):
    input_c = None
    output_c = None
    def __init__(self, modulus, fhe_builder: FheBuilder, data_range, img_hw, filter_hw,
                 num_input_channel, num_output_channel, name: str, padding=1):
        super().__init__(modulus, fhe_builder, data_range, img_hw, filter_hw,
                         num_input_channel, num_output_channel, name, Config.client_rank, padding)

        self.fhe_ntt = Conv2dFheNttClient(modulus, fhe_builder, data_range, img_hw, filter_hw,
                                          num_input_channel, num_output_channel, name, padding=padding)

    def offline(self, input_c):
        self.fhe_ntt.offline(input_c)
        self.output_c = self.fhe_ntt.output_c

    def online(self):
        pass


def test_conv2d_secure_comm(input_sid, master_address, master_port, setting=(16, 3, 128, 128)):
    test_name = "Conv2d Secure Comm"
    print(f"\nTest for {test_name}: Start")
    modulus = 786433
    padding = 1
    img_hw, filter_hw, num_input_channel, num_output_channel = setting
    data_bit = 17
    data_range = 2 ** data_bit
    # n_23 = 8192
    n_23 = 16384
    print(f"Setting covn2d: img_hw: {img_hw}, "
          f"filter_hw: {filter_hw}, "
          f"num_input_channel: {num_input_channel}, "
          f"num_output_channel: {num_output_channel}")

    x_shape = [num_input_channel, img_hw, img_hw]
    w_shape = [num_output_channel, num_input_channel, filter_hw, filter_hw]
    b_shape = [num_output_channel]

    fhe_builder = FheBuilder(modulus, n_23)
    fhe_builder.generate_keys()

    weight = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(w_shape)).reshape(w_shape)
    bias = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(b_shape)).reshape(b_shape)
    input = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(x_shape)).reshape(x_shape)
    input_c = generate_random_mask(modulus, x_shape)
    input_s = pmod(input - input_c, modulus)

    def check_correctness_online(x, w, b, output_s, output_c):
        actual = pmod(output_s.cuda() + output_c.cuda(), modulus)
        torch_x = x.reshape([1] + x_shape).cuda().double()
        torch_w = w.reshape(w_shape).cuda().double()
        torch_b = b.cuda().double() if b is not None else None
        expected = F.conv2d(torch_x, torch_w, padding=padding, bias=torch_b)
        expected = pmod(expected.reshape(output_s.shape), modulus)
        compare_expected_actual(expected, actual, name=test_name + " online", get_relative=True)

    def test_server():
        rank = Config.server_rank
        init_communicate(rank, master_address=master_address, master_port=master_port)
        warming_up_cuda()
        prot = Conv2dSecureServer(modulus, fhe_builder, data_range, img_hw, filter_hw,
                                  num_input_channel, num_output_channel, "test_conv2d_secure_comm", padding=padding)
        with NamedTimerInstance("Server Offline"):
            prot.offline(weight, bias=bias)
            torch_sync()
        with NamedTimerInstance("Server Online"):
            prot.online(input_s)
            torch_sync()

        blob_output_c = BlobTorch(prot.output_shape, torch.float, prot.comm_base, "output_c")
        blob_output_c.prepare_recv()
        torch_sync()
        output_c = blob_output_c.get_recv()
        check_correctness_online(input, weight, bias, prot.output_s, output_c)

        end_communicate()

    def test_client():
        rank = Config.client_rank
        init_communicate(rank, master_address=master_address, master_port=master_port)
        warming_up_cuda()
        prot = Conv2dSecureClient(modulus, fhe_builder, data_range, img_hw, filter_hw,
                                  num_input_channel, num_output_channel, "test_conv2d_secure_comm", padding=padding)
        with NamedTimerInstance("Client Offline"):
            prot.offline(input_c)
            torch_sync()
        with NamedTimerInstance("Client Online"):
            prot.online()
            torch_sync()

        blob_output_c = BlobTorch(prot.output_shape, torch.float, prot.comm_base, "output_c")
        torch_sync()
        blob_output_c.send(prot.output_c)
        end_communicate()

    if input_sid == Config.both_rank:
        marshal_funcs([test_server, test_client])
    elif input_sid == Config.server_rank:
        test_server()
    elif input_sid == Config.client_rank:
        test_client()

    print(f"\nTest for {test_name}: End")


if __name__ == "__main__":
    # test_ntt_conv()
    # test_rotation()
    # test_conv2d_fhe_ntt_single_thread()
    # test_conv2d_fhe_ntt_comm()
    input_sid, master_address, master_port, test_to_run = argparser_distributed()
    sys.stdout = Logger()
    num_repeat = 10
    try_setting = [(28, 5, 1, 5), (16, 3, 128, 128), (16, 3, 256, 256), (8, 3, 256, 256), (32, 3, 32, 32)]
    for _, setting in product(range(num_repeat), try_setting):
        test_conv2d_secure_comm(input_sid, master_address, master_port, setting=setting)