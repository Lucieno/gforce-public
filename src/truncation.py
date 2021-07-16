import sys
from itertools import product

import torch

from comm import CommBase, CommFheBuilder, NamedBase, BlobTorch, PhaseProtocolServer, PhaseProtocolClient, \
    init_communicate, torch_sync, end_communicate, PhaseProtocolCommon, BlobFheEnc, TrafficRecord
from config import Config
from fhe import FheBuilder
from logger_utils import Logger
from timer_utils import NamedTimerInstance
from torch_utils import generate_random_mask, pmod, gen_unirand_int_grain, compare_expected_actual, marshal_funcs, \
    warming_up_cuda, argparser_distributed, MetaTruncRandomGenerator


class TruncCommon(NamedBase, PhaseProtocolCommon):
    class_name = "Trunc"
    def __init__(self, num_elem, modulus, pow_to_div: int, fhe_builder: FheBuilder, name: str, rank):
        super().__init__(name)
        self.rank = rank
        self.name = name
        self.comm_base = CommBase(rank, self.sub_name("comm_base"))
        self.num_elem = num_elem
        self.modulus = modulus
        self.pow_to_div = pow_to_div
        self.div = 2 ** pow_to_div
        self.nullify_threshold = (modulus - 1) / 2
        self.fhe_builder = fhe_builder
        self.comm_fhe = CommFheBuilder(rank, fhe_builder, self.sub_name("comm_fhe"))
        self.comp_dev = torch.device("cuda")

        assert(self.fhe_builder.modulus == self.modulus)

        self.blob_pre_c = BlobFheEnc(self.num_elem, self.comm_fhe, "pre_c")
        self.blob_wrap_c = BlobFheEnc(self.num_elem, self.comm_fhe, "wrap_c")

        self.blob_sum_xr_s = BlobTorch(self.num_elem, torch.float, self.comm_base, "sum_xr_s", dst_device=self.comp_dev)
        self.blob_pre_s = BlobTorch(self.num_elem, torch.float, self.comm_base, "pre_s", dst_device=self.comp_dev)

        self.offline_server_send = [self.blob_wrap_c]
        self.offline_client_send = [self.blob_pre_c]
        self.online_server_send = [self.blob_sum_xr_s]
        self.online_client_send = [self.blob_pre_s]

    def generate_mask(self):
        return gen_unirand_int_grain(0, self.modulus - 1, self.num_elem).to(Config.device)

    def mod_to_modulus(self, x):
        return pmod(x, self.modulus)


class TruncServer(TruncCommon, PhaseProtocolServer):
    wrap_s: torch.Tensor
    out_s: torch.Tensor
    def __init__(self, num_elem, modulus, pow_to_div: int, fhe_builder: FheBuilder, name: str):
        super().__init__(num_elem, modulus, pow_to_div, fhe_builder, name, Config.server_rank)

    def offline(self):
        PhaseProtocolServer.offline(self)

        self.elem_zeros = torch.zeros(self.num_elem).to(Config.device)
        # self.r = self.generate_mask()
        # self.r = torch.zeros(self.num_elem).to(Config.device)
        # self.r = torch.arange(self.num_elem).to(Config.device)
        meta_rg = MetaTruncRandomGenerator()
        rg = meta_rg.get_rg("secure")
        self.r = rg.gen_uniform(self.num_elem, self.modulus).to(Config.device)
        self.mult = torch.where((self.r < self.nullify_threshold),
                                self.elem_zeros, self.elem_zeros + self.modulus//self.div).double()
        self.wrap_mask_s = self.generate_mask()
        fhe_mult = self.fhe_builder.build_plain_from_torch(self.mult)
        fhe_bias = self.fhe_builder.build_plain_from_torch(self.wrap_mask_s)
        fhe_pre_c = self.blob_pre_c.get_recv()
        fhe_pre_c *= fhe_mult
        fhe_pre_c += fhe_bias
        self.blob_wrap_c.send(fhe_pre_c)

    def online(self, input_s):
        sum_xr_s = self.mod_to_modulus(self.r + input_s.to(Config.device))
        self.blob_sum_xr_s.send(sum_xr_s)

        pre_s = self.blob_pre_s.get_recv()
        self.wrap_s = self.mult * pre_s - self.wrap_mask_s
        self.out_s = self.mod_to_modulus(-self.r//self.div + self.wrap_s)


class TruncClient(TruncCommon, PhaseProtocolClient):
    out_c: torch.Tensor
    def __init__(self, num_elem, modulus, pow_to_div: int, fhe_builder: FheBuilder, name: str):
        super().__init__(num_elem, modulus, pow_to_div, fhe_builder, name, Config.client_rank)

    def offline(self):
        PhaseProtocolClient.offline(self)

        self.elem_zeros = torch.zeros(self.num_elem).to(Config.device)
        self.pre_c = self.generate_mask()
        self.blob_pre_c.send_from_torch(self.pre_c)

        self.wrap_c = self.blob_wrap_c.get_recv_decrypt()

    def online(self, input_c):
        sum_xr_s = self.blob_sum_xr_s.get_recv()
        sum_xr = self.mod_to_modulus(sum_xr_s + input_c.to(Config.device))
        pre_s = torch.where(sum_xr < self.nullify_threshold, self.elem_zeros + 1, self.elem_zeros)
        pre_s = self.mod_to_modulus(pre_s - self.pre_c)
        self.blob_pre_s.send(pre_s)
        self.out_c = self.mod_to_modulus(sum_xr//self.div + self.wrap_c)


def test_trunc_comm(input_sid, master_address, master_port, num_elem=2**17):
    test_name = "test_trunc_comm"
    print(f"\nTest for {test_name}: Start")
    # modulus = 786433
    modulus = Config.q_23
    data_range = 2 ** 20
    pow_to_div = 10

    print(f"Number of element: {num_elem}")

    def check_correctness_online(img, out_s, out_c, pow_to_div):
        div = 2 ** pow_to_div
        expected = pmod(img//div, modulus).to(Config.device)
        actual = pmod(out_s + out_c, modulus).to(Config.device)
        err_indices = ((actual - expected) != 0) & ((actual - expected) != 1)
        unacceptable = torch.sum(err_indices).item()
        print("unacceptable:", unacceptable)

    def test_server():
        warming_up_cuda()
        rank = Config.server_rank
        init_communicate(rank, master_address=master_address, master_port=master_port)
        traffic_record = TrafficRecord()
        fhe_builder = FheBuilder(modulus, Config.n_23)
        comm_fhe = CommFheBuilder(rank, fhe_builder, "fhe_builder")
        torch_sync()
        comm_fhe.recv_public_key()
        comm_fhe.wait_and_build_public_key()

        x = gen_unirand_int_grain(0, data_range // 2, num_elem)
        x_s = gen_unirand_int_grain(0, modulus - 1, num_elem).cpu()
        x_c = pmod(x - x_s, modulus)

        prot = TruncServer(num_elem, modulus, pow_to_div, fhe_builder, test_name)
        blob_x_c = BlobTorch(num_elem, torch.float, prot.comm_base, "x_c")
        torch_sync()
        blob_x_c.send(x_c)

        torch_sync()
        with NamedTimerInstance("Server Offline"):
            prot.offline()
            torch_sync()
        traffic_record.reset("server-offline")

        with NamedTimerInstance("Server Online"):
            prot.online(x_s)
            torch_sync()
        traffic_record.reset("server-online")

        blob_output_c = BlobTorch(num_elem, torch.float, prot.comm_base, "output_c")
        blob_output_c.prepare_recv()
        torch_sync()
        output_c = blob_output_c.get_recv()
        check_correctness_online(x, prot.out_s, output_c, pow_to_div)

        end_communicate()

    def test_client():
        warming_up_cuda()
        rank = Config.client_rank
        init_communicate(rank, master_address=master_address, master_port=master_port)
        traffic_record = TrafficRecord()
        fhe_builder = FheBuilder(modulus, Config.n_23)
        fhe_builder.generate_keys()
        comm_fhe = CommFheBuilder(rank, fhe_builder, "fhe_builder")
        torch_sync()
        comm_fhe.send_public_key()

        prot = TruncClient(num_elem, modulus, pow_to_div, fhe_builder, test_name)

        blob_x_c = BlobTorch(num_elem, torch.float, prot.comm_base, "x_c")
        blob_x_c.prepare_recv()
        torch_sync()
        x_c = blob_x_c.get_recv()

        torch_sync()
        with NamedTimerInstance("Client Offline"):
            prot.offline()
            torch_sync()
        traffic_record.reset("client-offline")

        with NamedTimerInstance("Client Online"):
            prot.online(x_c)
            torch_sync()
        traffic_record.reset("client-online")

        blob_output_c = BlobTorch(num_elem, torch.float, prot.comm_base, "output_c")
        torch_sync()
        blob_output_c.send(prot.out_c)
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
        test_trunc_comm(input_sid, master_address, master_port, num_elem=num_elem)