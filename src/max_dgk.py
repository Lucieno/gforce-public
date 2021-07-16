import cProfile, pstats

import torch

from comm import CommFheBuilder, CommBase, PhaseProtocolServer, PhaseProtocolClient, BlobFheEnc, BlobTorch, \
    end_communicate, torch_sync, init_communicate
from config import Config
from dgk_basic import DgkBitServer, DgkBitClient
from dgk_basic import DgkCommBase
from fhe import FheBuilder
from shares_mult import SharesMultServer, SharesMultClient
from timer_utils import NamedTimerInstance
from torch_utils import gen_unirand_int_grain, pmod, marshal_funcs, compare_expected_actual


class MaxDgkComm(DgkCommBase):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str, rank):
        super(MaxDgkComm, self).__init__(num_elem, q_23, q_16, work_bit, data_bit,
                                         fhe_builder_16, fhe_builder_23, name, rank, "MaxDgkComm")
        self.fhe_builder = self.fhe_builder_23
        self.modulus = self.q_23


class MaxDgkServer(MaxDgkComm):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str):
        super(MaxDgkServer, self).__init__(num_elem, q_23, q_16, work_bit, data_bit,
                                           fhe_builder_16, fhe_builder_23, name, Config.server_rank)
        self.dgk = DgkBitServer(num_elem, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, name)
        self.shares_mult = SharesMultServer(num_elem, q_23, self.fhe_builder_23, name)

    def offline(self):
        self.dgk.offline()
        self.shares_mult.offline()

        self.multipled_mask = self.generate_random().to(Config.device)
        self.fhe_multipled_mask = self.fhe_builder.build_plain_from_torch(self.multipled_mask)

    def online(self, x_s, y_s):
        x_s = x_s.to(Config.device)
        y_s = y_s.to(Config.device)
        y_sub_x_s = self.mod_to_modulus(y_s - x_s).to(Config.device)

        self.dgk.online(y_sub_x_s)
        self.shares_mult.online(y_sub_x_s, self.dgk.dgk_x_leq_y_s)

        self.max_s = self.mod_to_modulus(self.shares_mult.c_s + x_s)


class MaxDgkClient(MaxDgkComm):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str):
        super(MaxDgkClient, self).__init__(num_elem, q_23, q_16, work_bit, data_bit,
                                           fhe_builder_16, fhe_builder_23, name, Config.client_rank)
        self.dgk = DgkBitClient(num_elem, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, name)
        self.shares_mult = SharesMultClient(num_elem, q_23, self.fhe_builder_23, name)

    def offline(self):
        self.dgk.offline()
        self.shares_mult.offline()

    def online(self, x_c, y_c):
        x_c = x_c.to(Config.device)
        y_c = y_c.to(Config.device)
        y_sub_x_c = self.mod_to_modulus(y_c - x_c).to(Config.device)

        self.dgk.online(y_sub_x_c)
        self.shares_mult.online(y_sub_x_c, self.dgk.dgk_x_leq_y_c)

        self.max_c = self.mod_to_modulus(self.shares_mult.c_c + x_c)


def test_max_dgk():
    test_name = "Max Dgk"
    print(f"\nTest for {test_name}: Start")
    data_bit = 17
    work_bit = 17
    data_range = 2 ** data_bit
    q_16 = 12289
    q_23 = 786433
    num_elem = 2 ** 17
    print(f"Number of element: {num_elem}")

    fhe_builder_16 = FheBuilder(q_16, 2048)
    fhe_builder_16.generate_keys()
    fhe_builder_23 = FheBuilder(q_23, 8192)
    fhe_builder_23.generate_keys()

    a = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem)
    b = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem)
    a_s = gen_unirand_int_grain(0, q_23 - 1, num_elem)
    a_c = pmod(a - a_s, q_23)
    b_s = gen_unirand_int_grain(0, q_23 - 1, num_elem)
    b_c = pmod(b - b_s, q_23)

    def check_correctness_online(a, b, c_s, c_c):
        expected = pmod(torch.max(a.to(Config.device), b.to(Config.device)), q_23)
        actual = pmod(c_s + c_c, q_23)
        compare_expected_actual(expected, actual, name="max_dgk_online", get_relative=True)

    def test_server():
        init_communicate(Config.server_rank)
        prot = MaxDgkServer(num_elem, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, "max_dgk")
        with NamedTimerInstance("Server Offline"):
            prot.offline()
            torch_sync()

        with NamedTimerInstance("Server Online"):
            prot.online(a_s, b_s)
            torch_sync()

        blob_max_c = BlobTorch(num_elem, torch.float, prot.comm_base, "recon_max_c")
        blob_max_c.prepare_recv()
        torch_sync()
        max_c = blob_max_c.get_recv()
        check_correctness_online(a, b, prot.max_s, max_c)

        end_communicate()

    def test_client():
        init_communicate(Config.client_rank)
        prot = MaxDgkClient(num_elem, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, "max_dgk")
        with NamedTimerInstance("Client Offline"):
            prot.offline()
            torch_sync()

        with NamedTimerInstance("Client Online"):
            prot.online(a_c, b_c)
            torch_sync()

        blob_max_c = BlobTorch(num_elem, torch.float, prot.comm_base, "recon_max_c")
        torch_sync()
        blob_max_c.send(prot.max_c)
        end_communicate()

    marshal_funcs([test_server, test_client])
    print(f"\nTest for {test_name}: End")

if __name__ == "__main__":
    test_max_dgk()
