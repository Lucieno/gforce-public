import sys
from itertools import product
from torch.multiprocessing import Process

import torch

from comm import CommFheBuilder, CommBase, PhaseProtocolServer, PhaseProtocolClient, BlobFheEnc, BlobTorch, \
    end_communicate, torch_sync, init_communicate, PhaseProtocolCommon, TrafficRecord
from config import Config
from dgk_basic import DgkBitServer, DgkBitClient, DgkCommBase
from dgk_single_thread import DgkBase
from fhe import FheBuilder
from logger_utils import Logger
from shares_mult import SharesMultServer, SharesMultClient
from timer_utils import NamedTimerInstance
from torch_utils import gen_unirand_int_grain, pmod, marshal_funcs, compare_expected_actual, argparser_distributed


class ReluDgkCommon(DgkCommBase, PhaseProtocolCommon):
    max_s: torch.Tensor
    max_c: torch.Tensor

    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str, rank):
        super(ReluDgkCommon, self).__init__(num_elem, q_23, q_16, work_bit, data_bit,
                                            fhe_builder_16, fhe_builder_23, name, rank, "ReluDgkComm")
        self.fhe_builder = self.fhe_builder_23
        self.modulus = self.q_23


class ReluDgkServer(ReluDgkCommon):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str):
        super(ReluDgkServer, self).__init__(num_elem, q_23, q_16, work_bit, data_bit,
                                            fhe_builder_16, fhe_builder_23, name, Config.server_rank)
        self.dgk = DgkBitServer(num_elem, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, name)
        self.shares_mult = SharesMultServer(num_elem, q_23, self.fhe_builder_23, name)

    def offline(self):
        self.dgk.offline()
        self.shares_mult.offline()

        self.multipled_mask = self.generate_random().to(Config.device)
        self.fhe_multipled_mask = self.fhe_builder.build_plain_from_torch(self.multipled_mask)

    def online(self, x_s):
        x_s = x_s.to(Config.device)

        self.dgk.online(x_s)
        self.shares_mult.online(x_s, self.dgk.dgk_x_leq_y_s)

        self.max_s = self.mod_to_modulus(self.shares_mult.c_s)


class ReluDgkClient(ReluDgkCommon):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str):
        super(ReluDgkClient, self).__init__(num_elem, q_23, q_16, work_bit, data_bit,
                                            fhe_builder_16, fhe_builder_23, name, Config.client_rank)
        self.dgk = DgkBitClient(num_elem, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, name)
        self.shares_mult = SharesMultClient(num_elem, q_23, self.fhe_builder_23, name)

    def offline(self):
        self.dgk.offline()
        self.shares_mult.offline()

    def online(self, x_c):
        x_c = x_c.to(Config.device)

        self.dgk.online(x_c)
        self.shares_mult.online(x_c, self.dgk.dgk_x_leq_y_c)

        self.max_c = self.mod_to_modulus(self.shares_mult.c_c)


def test_relu_dgk(input_sid, master_address, master_port, num_elem=2**17):
    test_name = "Relu Dgk"
    print(f"\nTest for {test_name}: Start")
    data_bit = 20
    work_bit = 20
    data_range = 2 ** data_bit
    # q_16 = 12289
    q_16 = Config.q_16
    # q_23 = 786433
    q_23 = Config.q_23
    # q_23 = 8273921
    print(f"Number of element: {num_elem}")

    def check_correctness_online(a, c_s, c_c):
        a = a.to(Config.device)
        expected = pmod(torch.max(a, torch.zeros_like(a)), q_23)
        actual = pmod(c_s + c_c, q_23)
        compare_expected_actual(expected, actual, name="relu_dgk_online", get_relative=True)

    def test_server():
        rank = Config.server_rank
        init_communicate(Config.server_rank, master_address=master_address, master_port=master_port)
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

        prot = ReluDgkServer(num_elem, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, "relu_dgk")

        blob_a_s = BlobTorch(num_elem, torch.float, prot.comm_base, "a")
        blob_max_s = BlobTorch(num_elem, torch.float, prot.comm_base, "max_s")
        torch_sync()
        blob_a_s.prepare_recv()
        a_s = blob_a_s.get_recv()

        torch_sync()
        with NamedTimerInstance("Server Offline"):
            prot.offline()
            torch_sync()
        traffic_record.reset("server-offline")

        with NamedTimerInstance("Server Online"):
            prot.online(a_s)
            torch_sync()
        traffic_record.reset("server-online")

        blob_max_s.send(prot.max_s)
        torch.cuda.empty_cache()
        end_communicate()

    def test_client():
        rank = Config.client_rank
        init_communicate(rank, master_address=master_address, master_port=master_port)
        traffic_record = TrafficRecord()
        fhe_builder_16 = FheBuilder(q_16, Config.n_16)
        fhe_builder_23 = FheBuilder(q_23, Config.n_23)
        fhe_builder_16.generate_keys()
        fhe_builder_23.generate_keys()
        comm_fhe_16 = CommFheBuilder(rank, fhe_builder_16, "fhe_builder_16")
        comm_fhe_23 = CommFheBuilder(rank, fhe_builder_23, "fhe_builder_23")
        torch_sync()
        comm_fhe_16.send_public_key()
        comm_fhe_23.send_public_key()

        a = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem)
        a_c = gen_unirand_int_grain(0, q_23 - 1, num_elem)
        a_s = pmod(a - a_c, q_23)

        prot = ReluDgkClient(num_elem, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, "relu_dgk")

        blob_a_s = BlobTorch(num_elem, torch.float, prot.comm_base, "a")
        blob_max_s = BlobTorch(num_elem, torch.float, prot.comm_base, "max_s")
        torch_sync()
        blob_a_s.send(a_s)
        blob_max_s.prepare_recv()

        torch_sync()
        with NamedTimerInstance("Client Offline"):
            prot.offline()
            torch_sync()
        traffic_record.reset("client-offline")

        with NamedTimerInstance("Client Online"):
            prot.online(a_c)
            torch_sync()
        traffic_record.reset("client-online")

        max_s = blob_max_s.get_recv()
        check_correctness_online(a, max_s, prot.max_c)

        torch.cuda.empty_cache()
        end_communicate()

    if input_sid == Config.both_rank:
        marshal_funcs([test_server, test_client])
    elif input_sid == Config.server_rank:
        marshal_funcs([test_server])
        # test_server()
    elif input_sid == Config.client_rank:
        marshal_funcs([test_client])
        # test_client()

    print(f"\nTest for {test_name}: End")

if __name__ == "__main__":
    import gc
    input_sid, master_address, master_port, test_to_run = argparser_distributed()
    sys.stdout = Logger()

    num_repeat = 10
    num_elem_try = [10000, 40000] + [2 ** i for i in range(10, 19)]

    for _, num_elem in product(range(num_repeat), num_elem_try):
        test_relu_dgk(input_sid, master_address, master_port, num_elem=num_elem)
        gc.collect()
