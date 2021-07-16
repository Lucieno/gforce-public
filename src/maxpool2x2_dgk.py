import sys
from itertools import product

import torch

from comm import CommFheBuilder, CommBase, PhaseProtocolServer, PhaseProtocolClient, BlobFheEnc, BlobTorch, \
    end_communicate, torch_sync, init_communicate, PhaseProtocolCommon, TrafficRecord
from config import Config
from dgk_basic import DgkBitServer, DgkBitClient, DgkCommBase
from dgk_single_thread import DgkBase
from fhe import FheBuilder
from logger_utils import Logger
from max_dgk import MaxDgkServer, MaxDgkClient
from shares_mult import SharesMultServer, SharesMultClient
from timer_utils import NamedTimerInstance
from torch_utils import gen_unirand_int_grain, pmod, marshal_funcs, compare_expected_actual, argparser_distributed


class Maxpool2x2DgkCommon(DgkCommBase, PhaseProtocolCommon):
    max_s: torch.Tensor
    max_c: torch.Tensor
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit, img_hw,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str, rank):
        super(Maxpool2x2DgkCommon, self).__init__(num_elem, q_23, q_16, work_bit, data_bit,
                                                  fhe_builder_16, fhe_builder_23, name, rank, "Maxpool2x2DgkComm")
        self.img_hw = img_hw
        self.fhe_builder = self.fhe_builder_23
        self.modulus = self.q_23

        if num_elem % 4 != 0:
            raise Exception(f"num_elem should be divisible by 4, but got {num_elem}")
        if img_hw % 2 != 0:
            raise Exception(f"img_hw should be divisible by 2, but got {img_hw}")

    def reorder_1_x(self, x: torch.Tensor) -> torch.Tensor:
        return x[::2] if x is not None else x

    def reorder_1_y(self, x: torch.Tensor) -> torch.Tensor:
        return x[1::2] if x is not None else x

    def reorder_2_x(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape([-1, self.img_hw//2])[::2, :].reshape(-1) if x is not None else x

    def reorder_2_y(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape([-1, self.img_hw//2])[1::2, :].reshape(-1) if x is not None else x


class Maxpool2x2DgkServer(Maxpool2x2DgkCommon):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit, img_hw,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str):
        super(Maxpool2x2DgkServer, self).__init__(num_elem, q_23, q_16, work_bit, data_bit, img_hw,
                                                  fhe_builder_16, fhe_builder_23, name, Config.server_rank)
        self.max_dgk_1 = MaxDgkServer(num_elem//2, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, name)
        self.max_dgk_2 = MaxDgkServer(num_elem//4, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, name)

    def offline(self):
        self.max_dgk_1.offline()
        self.max_dgk_2.offline()

    def online(self, img_s):
        img_1_x_s = self.reorder_1_x(img_s)
        img_1_y_s = self.reorder_1_y(img_s)

        self.max_dgk_1.online(img_1_x_s, img_1_y_s)
        img_2_s = self.max_dgk_1.max_s.float()
        img_2_x_s = self.reorder_2_x(img_2_s)
        img_2_y_s = self.reorder_2_y(img_2_s)

        self.max_dgk_2.online(img_2_x_s, img_2_y_s)
        self.max_s = self.max_dgk_2.max_s


class Maxpool2x2DgkClient(Maxpool2x2DgkCommon):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit, img_hw,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str):
        super(Maxpool2x2DgkClient, self).__init__(num_elem, q_23, q_16, work_bit, data_bit, img_hw,
                                                  fhe_builder_16, fhe_builder_23, name, Config.client_rank)
        self.max_dgk_1 = MaxDgkClient(num_elem//2, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, name)
        self.max_dgk_2 = MaxDgkClient(num_elem//4, q_23, q_16, work_bit, data_bit, fhe_builder_16, fhe_builder_23, name)

    def offline(self):
        self.max_dgk_1.offline()
        self.max_dgk_2.offline()

    def online(self, img_c):
        img_1_x_c = self.reorder_1_x(img_c)
        img_1_y_c = self.reorder_1_y(img_c)

        self.max_dgk_1.online(img_1_x_c, img_1_y_c)
        img_2_c = self.max_dgk_1.max_c.float()
        img_2_x_c = self.reorder_2_x(img_2_c)
        img_2_y_c = self.reorder_2_y(img_2_c)

        self.max_dgk_2.online(img_2_x_c, img_2_y_c)
        self.max_c = self.max_dgk_2.max_c


def test_maxpool2x2_dgk(input_sid, master_address, master_port, num_elem=2**17):
    test_name = "Maxpool2x2 Dgk"
    print(f"\nTest for {test_name}: Start")
    data_bit = 20
    work_bit = 20
    data_range = 2 ** data_bit
    q_16 = Config.q_16
    # q_23 = 786433
    q_23 = Config.q_23
    img_hw = 4
    print(f"Number of element: {num_elem}")

    def check_correctness_online(img, max_s, max_c):
        img = torch.where(img < q_23 // 2, img, img - q_23).to(Config.device)
        pool = torch.nn.MaxPool2d(2)
        expected = pool(img.reshape([-1, img_hw, img_hw])).reshape(-1)
        expected = pmod(expected, q_23)
        actual = pmod(max_s + max_c, q_23)
        compare_expected_actual(expected, actual, name="maxpool2x2_dgk_online", get_relative=True)

    def test_server():
        rank = Config.server_rank
        init_communicate(rank, master_address=master_address, master_port=master_port)
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

        img = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem)
        img_s = gen_unirand_int_grain(0, q_23 - 1, num_elem)
        img_c = pmod(img - img_s, q_23)

        prot = Maxpool2x2DgkServer(num_elem, q_23, q_16, work_bit, data_bit, img_hw, fhe_builder_16, fhe_builder_23, "max_dgk")

        blob_img_c = BlobTorch(num_elem, torch.float, prot.comm_base, "recon_max_c")
        torch_sync()
        blob_img_c.send(img_c)

        torch_sync()
        with NamedTimerInstance("Server Offline"):
            prot.offline()
            torch_sync()
        traffic_record.reset("server-offline")

        with NamedTimerInstance("Server Online"):
            prot.online(img_s)
            torch_sync()
        traffic_record.reset("server-online")

        blob_max_c = BlobTorch(num_elem//4, torch.float, prot.comm_base, "recon_max_c")
        blob_max_c.prepare_recv()
        torch_sync()
        max_c = blob_max_c.get_recv()
        check_correctness_online(img, prot.max_s, max_c)

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

        prot = Maxpool2x2DgkClient(num_elem, q_23, q_16, work_bit, data_bit, img_hw, fhe_builder_16, fhe_builder_23, "max_dgk")
        blob_img_c = BlobTorch(num_elem, torch.float, prot.comm_base, "recon_max_c")
        blob_img_c.prepare_recv()
        torch_sync()
        img_c = blob_img_c.get_recv()

        torch_sync()
        with NamedTimerInstance("Client Offline"):
            prot.offline()
            torch_sync()
        traffic_record.reset("client-offline")

        with NamedTimerInstance("Client Online"):
            prot.online(img_c)
            torch_sync()
        traffic_record.reset("client-online")

        blob_max_c = BlobTorch(num_elem//4, torch.float, prot.comm_base, "recon_max_c")
        torch_sync()
        blob_max_c.send(prot.max_c)
        end_communicate()

    if input_sid == Config.both_rank:
        marshal_funcs([test_server, test_client])
    elif input_sid == Config.server_rank:
        marshal_funcs([test_server])
    elif input_sid == Config.client_rank:
        marshal_funcs([test_client])

    print(f"\nTest for {test_name}: End")


if __name__ == "__main__":
    input_sid, master_address, master_port, test_to_run = argparser_distributed()
    sys.stdout = Logger()
    num_repeat = 10
    num_elem_try = [10000, 40000] + [2 ** i for i in range(10, 19)]
    for _, num_elem in product(range(num_repeat), num_elem_try):
        test_maxpool2x2_dgk(input_sid, master_address, master_port, num_elem=num_elem)
