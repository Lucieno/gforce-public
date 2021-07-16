import sys
from itertools import product

import torch

from comm import CommFheBuilder, CommBase, PhaseProtocolServer, PhaseProtocolClient, BlobFheEnc, BlobTorch, \
    end_communicate, torch_sync, init_communicate, PhaseProtocolCommon, TrafficRecord
from config import Config
from dgk_basic import DgkBitServer, DgkBitClient, DgkCommBase
from fhe import FheBuilder
from logger_utils import Logger
from max_dgk import MaxDgkServer, MaxDgkClient
from timer_utils import NamedTimerInstance
from torch_utils import gen_unirand_int_grain, pmod, marshal_funcs, compare_expected_actual, argparser_distributed, \
    warming_up_cuda


class Avgpool2x2Common(DgkCommBase, PhaseProtocolCommon):
    out_s: torch.Tensor
    max_c: torch.Tensor
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit, img_hw,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str, rank):
        super(Avgpool2x2Common, self).__init__(num_elem, q_23, q_16, work_bit, data_bit,
                                               fhe_builder_16, fhe_builder_23, name, rank, "Avgpool2x2Comm")
        self.img_hw = img_hw
        self.fhe_builder = self.fhe_builder_23
        self.modulus = self.q_23
        self.pool = torch.nn.AvgPool2d(2, divisor_override=1)

        if num_elem % 4 != 0:
            raise Exception(f"num_elem should be divisible by 4, but got {num_elem}")
        if img_hw % 2 != 0:
            raise Exception(f"img_hw should be divisible by 2, but got {img_hw}")

        self.blob_offline_input = BlobTorch(num_elem, torch.float, self.comm_base, "offline_input")
        self.blob_online_input = BlobTorch(num_elem // 4, torch.float, self.comm_base, "online_input")

    def pooling(self, img):
        return self.pool(img.reshape([-1, self.img_hw, self.img_hw])).reshape(-1)

class Avgpool2x2Server(Avgpool2x2Common):
    out_s: torch.Tensor
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit, img_hw,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str):
        super(Avgpool2x2Server, self).__init__(num_elem, q_23, q_16, work_bit, data_bit, img_hw,
                                               fhe_builder_16, fhe_builder_23, name, Config.server_rank)

    def offline(self):
        pass

    def online(self, img_s):
        img_s = img_s.cuda().double()
        self.out_s = self.pooling(img_s)


class Avgpool2x2Client(Avgpool2x2Common):
    def __init__(self, num_elem, q_23, q_16, work_bit, data_bit, img_hw,
                 fhe_builder_16: FheBuilder, fhe_builder_23: FheBuilder, name: str):
        super(Avgpool2x2Client, self).__init__(num_elem, q_23, q_16, work_bit, data_bit, img_hw,
                                               fhe_builder_16, fhe_builder_23, name, Config.client_rank)

    def offline(self):
        pass

    def online(self, img_c):
        img_c = img_c.cuda().double()
        self.out_c = self.pooling(img_c)


def test_avgpool2x2_dgk(input_sid, master_address, master_port, num_elem=2**17):
    test_name = "Avgpool2x2"
    print(f"\nTest for {test_name}: Start")
    data_bit = 20
    work_bit = 20
    data_range = 2 ** data_bit
    q_16 = 12289
    # q_23 = 786433
    q_23 = 7340033
    img_hw = 4
    print(f"Number of element: {num_elem}")

    fhe_builder_16 = FheBuilder(q_16, Config.n_16)
    fhe_builder_16.generate_keys()
    fhe_builder_23 = FheBuilder(q_23, Config.n_23)
    fhe_builder_23.generate_keys()

    img = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, num_elem)
    img_s = gen_unirand_int_grain(0, q_23 - 1, num_elem)
    img_c = pmod(img - img_s, q_23)

    def check_correctness_online(img, max_s, max_c):
        img = torch.where(img < q_23 // 2, img, img - q_23).cuda()
        pool = torch.nn.AvgPool2d(2)
        expected = pool(img.double().reshape([-1, img_hw, img_hw])).reshape(-1) * 4
        expected = pmod(expected, q_23)
        actual = pmod(max_s + max_c, q_23)
        compare_expected_actual(expected, actual, name=test_name+"_online", get_relative=True)

    def test_server():
        init_communicate(Config.server_rank, master_address=master_address, master_port=master_port)
        warming_up_cuda()
        traffic_record = TrafficRecord()
        prot = Avgpool2x2Server(num_elem, q_23, q_16, work_bit, data_bit, img_hw, fhe_builder_16, fhe_builder_23, "avgpool")

        with NamedTimerInstance("Server Offline"):
            prot.offline()
            torch_sync()
        traffic_record.reset("server-offline")

        with NamedTimerInstance("Server Online"):
            prot.online(img_s)
            torch_sync()
        traffic_record.reset("server-online")

        blob_out_c = BlobTorch(num_elem//4, torch.float, prot.comm_base, "recon_res_c")
        blob_out_c.prepare_recv()
        torch_sync()
        out_c = blob_out_c.get_recv()
        check_correctness_online(img, prot.out_s, out_c)

        end_communicate()

    def test_client():
        init_communicate(Config.client_rank, master_address=master_address, master_port=master_port)
        warming_up_cuda()
        traffic_record = TrafficRecord()
        prot = Avgpool2x2Client(num_elem, q_23, q_16, work_bit, data_bit, img_hw, fhe_builder_16, fhe_builder_23, "avgpool")

        with NamedTimerInstance("Client Offline"):
            prot.offline()
            torch_sync()
        traffic_record.reset("client-offline")

        with NamedTimerInstance("Client Online"):
            prot.online(img_c)
            torch_sync()
        traffic_record.reset("client-online")

        blob_out_c = BlobTorch(num_elem//4, torch.float, prot.comm_base, "recon_res_c")
        torch_sync()
        blob_out_c.send(prot.out_c)
        end_communicate()

    if input_sid == Config.both_rank:
        marshal_funcs([test_server, test_client])
    elif input_sid == Config.server_rank:
        test_server()
    elif input_sid == Config.client_rank:
        test_client()

    print(f"\nTest for {test_name}: End")


if __name__ == "__main__":
    input_sid, master_address, master_port, test_to_run = argparser_distributed()
    sys.stdout = Logger()
    num_repeat = 1
    # num_elem_try = [10000, 40000] + [2 ** i for i in range(10, 21)]
    num_elem_try = [2 ** 14]
    for _, num_elem in product(range(num_repeat), num_elem_try):
        test_avgpool2x2_dgk(input_sid, master_address, master_port, num_elem=num_elem)
