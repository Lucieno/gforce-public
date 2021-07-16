import sys
from itertools import product
from math import ceil, floor

import torch
import numpy as np
from seal import uIntVector, Plaintext, Ciphertext

from comm import CommBase, CommFheBuilder, BlobFheRawCts, PhaseProtocolServer, PhaseProtocolClient, init_communicate, \
    torch_sync, end_communicate, BlobTorch, PhaseProtocolCommon
from config import Config
from fhe import FheBuilder, encrypt_zeros
from logger_utils import Logger
from timer_utils import NamedTimerInstance
from torch_utils import pmod, get_prod, gen_unirand_int_grain, warming_up_cuda, compare_expected_actual, \
    generate_random_mask, marshal_funcs, argparser_distributed


def calc_num_input_output_vec(num_elem_in_piece, num_input_unit, num_output_unit, degree):
    num_rot = int(ceil(num_input_unit / num_elem_in_piece))
    num_piece_in_vec = int(floor(degree / num_elem_in_piece))
    num_input_vec = num_rot
    num_output_vec = int(ceil(num_output_unit / num_piece_in_vec))
    return num_input_vec, num_output_vec


def calc_total_send(num_elem_in_piece, num_input_unit, num_output_unit, degree):
    a, b = calc_num_input_output_vec(num_elem_in_piece, num_input_unit, num_output_unit, degree)
    return a + b


def calc_total_compute(num_elem_in_piece, num_input_unit, num_output_unit, degree):
    a, b = calc_num_input_output_vec(num_elem_in_piece, num_input_unit, num_output_unit, degree)
    return a * b


def test_optimizing_unit():
    num_input_unit = 512
    num_output_unit = 512
    degree = 8192
    get_compute = lambda x: calc_total_compute(x, num_input_unit, num_output_unit, degree)
    get_send = lambda x: calc_total_send(x, num_input_unit, num_output_unit, degree)
    with NamedTimerInstance("Finding optimal args"):
        min_compute = min(range(1, num_input_unit+1), key=get_compute)
    min_send = min(range(1, num_input_unit+1), key=get_send)
    print(f"min_compute: {min_compute}, compute: {get_compute(min_compute)}, send: {get_send(min_compute)}")
    print(f"min_send: {min_send}, compute: {get_compute(min_send)}, send: {get_send(min_send)}")


class FcFheParam(object):
    class_name = "FcParam"
    def __init__(self, modulus, data_range, num_input_unit, num_output_unit, name):
        self.modulus = modulus
        self.data_range = data_range
        self.num_input_unit = num_input_unit
        self.num_output_unit = num_output_unit
        self.name = name
        self.degree = Config.n_23

        self.input_shape = torch.Size([self.num_input_unit])
        self.weight_shape = torch.Size([self.num_output_unit, self.num_input_unit])
        self.output_shape = torch.Size([self.num_output_unit])

        self.num_elem_in_piece = self.calc_optimal_num_elem_in_piece()
        self.num_rotation = int(ceil(self.num_input_unit / self.num_elem_in_piece))
        self.num_input_batch = self.num_rotation
        self.num_piece_in_batch = int(floor(self.degree / self.num_elem_in_piece))
        self.num_output_batch = int(ceil(self.num_output_unit / self.num_piece_in_batch))

    def calc_optimal_num_elem_in_piece(self):
        get_send = lambda x: calc_total_send(x, self.num_input_unit, self.num_output_unit, self.degree)
        min_send = min(range(1, self.num_input_unit), key=get_send)
        return min_send

    def sub_name(self, sub_name: str) -> str:
        return self.name + '_' + self.class_name + '_' + sub_name


class FcFheSingleThread(FcFheParam):
    class_name = "FcFheSingleThread"
    input_cts = None
    output_cts = None
    output_mask_s = None
    def __init__(self, modulus, data_range, num_input_unit, num_output_unit, fhe_builder: FheBuilder, name):
        super().__init__(modulus, data_range, num_input_unit, num_output_unit, name)
        self.fhe_builder = fhe_builder

        self.batch_encoder = self.fhe_builder.batch_encoder
        self.encryptor = self.fhe_builder.encryptor
        self.evaluator = self.fhe_builder.evaluator
        self.decryptor = self.fhe_builder.decryptor

        assert(self.degree == self.fhe_builder.degree)

    def index_input_batch_to_units(self, idx_batch):
        padded_span = self.num_elem_in_piece
        start = idx_batch * padded_span
        data_span = min(self.num_input_unit - start, padded_span)
        return start, start + data_span

    def index_output_batch_to_units(self, idx_batch, idx_piece):
        idx_output_unit = idx_batch * self.num_piece_in_batch + idx_piece
        if idx_output_unit >= self.num_output_unit:
            return False
        return idx_output_unit

    def index_weight_batch_to_units(self, idx_output_batch, idx_input_batch, idx_piece):
        idx_row = self.index_output_batch_to_units(idx_output_batch, idx_piece)
        idx_col_start, idx_col_end = self.index_input_batch_to_units(idx_input_batch)
        return idx_row, idx_col_start, idx_col_end

    def encode_input_to_fhe_batch(self, input_tensor):
        assert(input_tensor.shape == self.input_shape)
        self.input_cts = [Ciphertext() for _ in range(self.num_input_batch)]
        pod_vector = uIntVector()
        pt = Plaintext()
        for index_batch in range(self.num_input_batch):
            encoding_tensor = torch.zeros(self.degree)
            start_unit, end_unit = self.index_input_batch_to_units(index_batch)
            input_span = end_unit - start_unit
            piece_span = self.num_elem_in_piece
            for i in range(self.num_piece_in_batch):
                encoding_tensor[i*piece_span:i*piece_span+input_span] = input_tensor[start_unit:end_unit]
            encoding_tensor = pmod(encoding_tensor, self.modulus)
            pod_vector.from_np(encoding_tensor.numpy().astype(np.uint64))
            self.batch_encoder.encode(pod_vector, pt)
            self.encryptor.encrypt(pt, self.input_cts[index_batch])

    def compute_with_weight(self, weight_tensor):
        assert(weight_tensor.shape == self.weight_shape)
        pod_vector = uIntVector()
        pt = Plaintext()
        self.output_cts = encrypt_zeros(self.num_output_batch, self.batch_encoder, self.encryptor, self.degree)
        for idx_output_batch, idx_input_batch in product(range(self.num_output_batch), range(self.num_input_batch)):
            encoding_tensor = torch.zeros(self.degree)
            is_w_changed = False
            for idx_piece in range(self.num_piece_in_batch):
                idx_row, idx_col_start, idx_col_end = \
                    self.index_weight_batch_to_units(idx_output_batch, idx_input_batch, idx_piece)
                if idx_row is False:
                    continue
                is_w_changed = True
                padded_span = self.num_elem_in_piece
                data_span = idx_col_end - idx_col_start
                start_piece = idx_piece * padded_span
                encoding_tensor[start_piece: start_piece+data_span] = weight_tensor[idx_row, idx_col_start:idx_col_end]
            if not is_w_changed:
                continue
            encoding_tensor = pmod(encoding_tensor, self.modulus)
            pod_vector.from_np(encoding_tensor.numpy().astype(np.uint64))
            self.batch_encoder.encode(pod_vector, pt)
            sub_dotted = Ciphertext(self.input_cts[idx_input_batch])
            self.evaluator.multiply_plain_inplace(sub_dotted, pt)
            self.evaluator.add_inplace(self.output_cts[idx_output_batch], sub_dotted)

    def masking_output(self):
        spread_mask = generate_random_mask(self.modulus, [self.num_output_batch, self.degree])
        self.output_mask_s = torch.zeros(self.num_output_unit).double()

        pod_vector = uIntVector()
        pt = Plaintext()
        for idx_output_batch in range(self.num_output_batch):
            encoding_tensor = torch.zeros(self.degree, dtype=torch.float)
            for idx_piece in range(self.num_piece_in_batch):
                idx_output_unit = self.index_output_batch_to_units(idx_output_batch, idx_piece)
                if idx_output_unit is False:
                    break
                padded_span = self.num_elem_in_piece
                start_piece = idx_piece * padded_span
                arr = spread_mask[idx_output_batch, start_piece: start_piece+padded_span]
                encoding_tensor[start_piece: start_piece+padded_span] = arr
                self.output_mask_s[idx_output_unit] = arr.double().sum()
            encoding_tensor = pmod(encoding_tensor, self.modulus)
            pod_vector.from_np(encoding_tensor.numpy().astype(np.uint64))
            self.batch_encoder.encode(pod_vector, pt)
            self.evaluator.add_plain_inplace(self.output_cts[idx_output_batch], pt)

        self.output_mask_s = pmod(self.output_mask_s, self.modulus)

    def decode_output_from_fhe_batch(self):
        output_tensor = torch.zeros(self.output_shape).double()
        pod_vector = uIntVector()
        pt = Plaintext()
        cts = self.output_cts
        for idx_output_batch in range(self.num_output_batch):
            self.decryptor.decrypt(cts[idx_output_batch], pt)
            self.batch_encoder.decode(pt, pod_vector)
            arr = np.array(pod_vector, copy=False)
            arr = torch.from_numpy(arr.astype(np.double))
            for idx_piece in range(self.num_piece_in_batch):
                idx_output_unit = self.index_output_batch_to_units(idx_output_batch, idx_piece)
                if idx_output_unit is False:
                    break
                padded_span = self.num_elem_in_piece
                start_piece = idx_piece * padded_span
                output_tensor[idx_output_unit] = arr[start_piece: start_piece+padded_span].double().sum()

        output_tensor = pmod(output_tensor, self.modulus)
        return output_tensor


def test_fc_fhe_single_thread():
    test_name = "test_fc_fhe_single_thread"
    print(f"\nTest for {test_name}: Start")
    modulus = Config.q_23
    num_input_unit = 512
    num_output_unit = 512
    data_bit = 17

    data_range = 2 ** data_bit
    x_shape = [num_input_unit]
    w_shape = [num_output_unit, num_input_unit]

    fhe_builder = FheBuilder(modulus, Config.n_23)
    fhe_builder.generate_keys()

    # x = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(x_shape)).reshape(x_shape)
    w = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(w_shape)).reshape(w_shape)
    x = gen_unirand_int_grain(0, modulus, get_prod(x_shape)).reshape(x_shape)
    # w = gen_unirand_int_grain(0, modulus, get_prod(w_shape)).reshape(w_shape)
    # x = torch.arange(get_prod(x_shape)).reshape(x_shape)
    # w = torch.arange(get_prod(w_shape)).reshape(w_shape)

    warming_up_cuda()
    prot = FcFheSingleThread(modulus, data_range, num_input_unit, num_output_unit, fhe_builder, test_name)

    print("prot.num_input_batch", prot.num_input_batch)
    print("prot.num_output_batch", prot.num_output_batch)
    print("prot.num_elem_in_piece", prot.num_elem_in_piece)

    with NamedTimerInstance("encoding x"):
        prot.encode_input_to_fhe_batch(x)
    with NamedTimerInstance("conv2d with w"):
        prot.compute_with_weight(w)
    with NamedTimerInstance("conv2d masking output"):
        prot.masking_output()
    with NamedTimerInstance("decoding output"):
        y = prot.decode_output_from_fhe_batch()
    actual = pmod(y, modulus)
    actual = pmod(y - prot.output_mask_s, modulus)
    # print("actual\n", actual)

    torch_x = x.reshape([1] + x_shape).double()
    torch_w = w.reshape(w_shape).double()
    with NamedTimerInstance("Conv2d Torch"):
        expected = torch.mm(torch_x, torch_w.t())
        expected = pmod(expected.reshape(prot.output_shape), modulus)
    compare_expected_actual(expected, actual, name=test_name, get_relative=True)
    print(f"\nTest for {test_name}: End")


class FcFheCommon(FcFheParam):
    class_name = "FcFhe"
    def __init__(self, modulus, data_range, num_input_unit, num_output_unit, fhe_builder: FheBuilder, name, rank):
        super().__init__(modulus, data_range, num_input_unit, num_output_unit, name)

        self.fhe_builder = fhe_builder
        self.rank = rank
        self.comm_base = CommBase(rank, self.sub_name("comm_base"))
        self.comm_fhe = CommFheBuilder(rank, fhe_builder, self.sub_name("comm_fhe"))
        self.compute_core = FcFheSingleThread(modulus, data_range, num_input_unit, num_output_unit, fhe_builder, name)

        self.blob_input_cts = BlobFheRawCts(self.num_input_batch, self.comm_fhe, self.sub_name("input_cts"))
        self.blob_output_cts = BlobFheRawCts(self.num_output_batch, self.comm_fhe, self.sub_name("output_cts"))

        self.offline_server_send = [self.blob_output_cts]
        self.offline_client_send = [self.blob_input_cts]
        self.online_server_send = []
        self.online_client_send = []


class FcFheServer(FcFheCommon, PhaseProtocolServer):
    def __init__(self, modulus, data_range, num_input_unit, num_output_unit, fhe_builder, name):
        super().__init__(modulus, data_range, num_input_unit, num_output_unit, fhe_builder, name, Config.server_rank)

    def offline(self, weight_tensor):
        PhaseProtocolServer.offline(self)
        self.compute_core.input_cts = self.blob_input_cts.get_recv()
        self.compute_core.compute_with_weight(weight_tensor)
        self.compute_core.masking_output()
        self.blob_output_cts.send(self.compute_core.output_cts)
        self.output_mask_s = self.compute_core.output_mask_s.cuda().double()


class FcFheClient(FcFheCommon, PhaseProtocolClient):
    def __init__(self, modulus, data_range, num_input_unit, num_output_unit, fhe_builder, name):
        super().__init__(modulus, data_range, num_input_unit, num_output_unit, fhe_builder, name, Config.client_rank)

    def offline(self, input_mask):
        PhaseProtocolServer.offline(self)
        self.compute_core.encode_input_to_fhe_batch(input_mask)
        self.blob_input_cts.send(self.compute_core.input_cts)
        self.compute_core.output_cts = self.blob_output_cts.get_recv()
        self.output_c = self.compute_core.decode_output_from_fhe_batch().cuda().double()


def test_fc_fhe_comm():
    test_name = "test_fc_fhe_comm"
    print(f"\nTest for {test_name}: Start")
    modulus = 786433
    num_input_unit = 512
    num_output_unit = 512
    data_bit = 17

    print(f"Setting: num_input_unit {num_input_unit}, "
          f"num_output_unit: {num_output_unit}")

    data_range = 2 ** data_bit
    x_shape = [num_input_unit]
    w_shape = [num_output_unit, num_input_unit]

    fhe_builder = FheBuilder(modulus, Config.n_23)
    fhe_builder.generate_keys()

    weight = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(w_shape)).reshape(w_shape)
    input_mask = gen_unirand_int_grain(0, modulus - 1, get_prod(x_shape)).reshape(x_shape)
    # input_mask = torch.arange(get_prod(x_shape)).reshape(x_shape)

    def check_correctness_offline(x, w, output_mask, output_c):
        actual = pmod(output_c.cuda() - output_mask.cuda(), modulus)
        torch_x = x.reshape([1] + x_shape).cuda().double()
        torch_w = w.reshape(w_shape).cuda().double()
        expected = torch.mm(torch_x, torch_w.t())
        expected = pmod(expected.reshape(output_mask.shape), modulus)
        compare_expected_actual(expected, actual, name=test_name + " offline", get_relative=True)

    def test_server():
        init_communicate(Config.server_rank)
        prot = FcFheServer(modulus, data_range, num_input_unit, num_output_unit, fhe_builder, test_name)
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
        prot = FcFheClient(modulus, data_range, num_input_unit, num_output_unit, fhe_builder, test_name)
        with NamedTimerInstance("Client Offline"):
            prot.offline(input_mask)
            torch_sync()

        blob_output_c = BlobTorch(prot.output_shape, torch.float, prot.comm_base, "output_c")
        torch_sync()
        blob_output_c.send(prot.output_c)
        end_communicate()

    marshal_funcs([test_server, test_client])
    print(f"\nTest for {test_name}: End")


class FcSecureCommon(FcFheParam, PhaseProtocolCommon):
    class_name = "Conv2dSecure"
    def __init__(self, modulus, data_range, num_input_unit, num_output_unit, fhe_builder: FheBuilder, name, rank):
        super().__init__(modulus, data_range, num_input_unit, num_output_unit, name)
        self.fhe_builder = fhe_builder
        self.rank = rank
        self.comm_base = CommBase(rank, self.sub_name("comm_base"))
        self.comm_fhe = CommFheBuilder(rank, fhe_builder, self.sub_name("comm_fhe"))

        assert(self.fhe_builder.degree == self.degree)

        self.offline_server_send = []
        self.offline_client_send = []
        self.online_server_send = []
        self.online_client_send = []


class FcSecureServer(FcSecureCommon):
    weight = None
    bias = None
    output_mask_s = None
    output_s = None
    torch_w = None
    def __init__(self, modulus, data_range, num_input_unit, num_output_unit, fhe_builder: FheBuilder, name):
        super().__init__(modulus, data_range, num_input_unit, num_output_unit, fhe_builder, name, Config.server_rank)
        self.offline_core = FcFheServer(modulus, data_range, num_input_unit, num_output_unit, fhe_builder, name)

    def offline(self, weight, bias):
        self.weight = weight.cuda().double()
        self.bias = bias.cuda().double() if bias is not None else None
        self.offline_core.offline(weight)
        self.output_mask_s = self.offline_core.output_mask_s.cuda().double()
        self.torch_w = weight.reshape(self.weight_shape).cuda().double()

        # warming up
        warm_up_x = generate_random_mask(self.modulus, [1] + list(self.input_shape)).cuda().double()
        warm_up_y = torch.mm(warm_up_x, self.weight.t())

    def online(self, input_s):
        input_s = input_s.reshape([1] + list(self.input_shape)).cuda().double()
        y_s = torch.mm(input_s, self.weight.t())
        if self.bias is not None:
            y_s += self.bias
        y_s = pmod(y_s.reshape(self.output_shape) - self.output_mask_s , self.modulus)
        self.output_s = y_s


class FcSecureClient(FcSecureCommon):
    input_c = None
    output_c = None
    def __init__(self, modulus, data_range, num_input_unit, num_output_unit, fhe_builder: FheBuilder, name):
        super().__init__(modulus, data_range, num_input_unit, num_output_unit, fhe_builder, name, Config.client_rank)
        self.offline_core = FcFheClient(modulus, data_range, num_input_unit, num_output_unit, fhe_builder, name)

    def offline(self, input_c):
        self.offline_core.offline(input_c)
        self.output_c = self.offline_core.output_c

    def online(self):
        pass


def test_fc_secure_comm(input_sid, master_address, master_port, setting=(512, 512)):
    test_name = "test_fc_secure_comm"
    print(f"\nTest for {test_name}: Start")
    modulus = 786433
    num_input_unit, num_output_unit = setting
    data_bit = 17

    print(f"Setting fc: "
          f"num_input_unit: {num_input_unit}, "
          f"num_output_unit: {num_output_unit}")

    data_range = 2 ** data_bit
    x_shape = [num_input_unit]
    w_shape = [num_output_unit, num_input_unit]
    b_shape = [num_output_unit]

    fhe_builder = FheBuilder(modulus, Config.n_23)
    fhe_builder.generate_keys()

    weight = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(w_shape)).reshape(w_shape)
    bias = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(b_shape)).reshape(b_shape)
    input = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, get_prod(x_shape)).reshape(x_shape)
    input_c = generate_random_mask(modulus, x_shape)
    input_s = pmod(input - input_c, modulus)

    def check_correctness_online(x, w, output_s, output_c):
        actual = pmod(output_s.cuda() + output_c.cuda(), modulus)
        torch_x = x.reshape([1] + x_shape).cuda().double()
        torch_w = w.reshape(w_shape).cuda().double()
        expected = torch.mm(torch_x, torch_w.t())
        if bias is not None:
            expected += bias.cuda().double()
        expected = pmod(expected.reshape(output_s.shape), modulus)
        compare_expected_actual(expected, actual, name=test_name + " online", get_relative=True)

    def test_server():
        rank = Config.server_rank
        init_communicate(rank, master_address=master_address, master_port=master_port)
        warming_up_cuda()
        prot = FcSecureServer(modulus, data_range, num_input_unit, num_output_unit, fhe_builder, test_name)
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
        check_correctness_online(input, weight, prot.output_s, output_c)

        end_communicate()

    def test_client():
        rank = Config.client_rank
        init_communicate(rank, master_address=master_address, master_port=master_port)
        warming_up_cuda()
        prot = FcSecureClient(modulus, data_range, num_input_unit, num_output_unit, fhe_builder, test_name)
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

    marshal_funcs([test_server, test_client])
    print(f"\nTest for {test_name}: End")



if __name__ == "__main__":
    # test_optimizing_unit()
    # test_fc_fhe_single_thread()
    # test_fc_fhe_comm()
    input_sid, master_address, master_port, test_to_run = argparser_distributed()
    sys.stdout = Logger()
    num_repeat = 5
    try_setting = [(2048, 1), (1024, 16), (1024, 128), (512, 512)]
    for _, setting in product(range(num_repeat), try_setting):
        test_fc_secure_comm(input_sid, master_address, master_port, setting=setting)
