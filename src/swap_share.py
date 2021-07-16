import torch

from comm import CommBase, CommFheBuilder, NamedBase, BlobTorch, PhaseProtocolServer, PhaseProtocolClient, \
    init_communicate, torch_sync, end_communicate, PhaseProtocolCommon
from config import Config
from fhe import FheBuilder
from timer_utils import NamedTimerInstance
from torch_utils import generate_random_mask, pmod, gen_unirand_int_grain, compare_expected_actual, marshal_funcs, \
    warming_up_cuda


class SwapToClientOfflineCommon(NamedBase, PhaseProtocolCommon):
    class_name = "SwapToClient"
    def __init__(self, num_elem, modulus, name: str, rank):
        super().__init__(name)
        self.rank = rank
        self.name = name
        self.comm_base = CommBase(rank, self.sub_name("comm_base"))
        self.num_elem = num_elem
        self.modulus = modulus
        self.comp_device = torch.device("cuda")

        self.blob_masked_input_s = BlobTorch(self.num_elem, torch.float, self.comm_base, "masked_input_s", dst_device=self.comp_device)
        self.blob_masked_output_s = BlobTorch(self.num_elem, torch.float, self.comm_base, "masked_output_s", dst_device=self.comp_device)

        self.offline_server_send = []
        self.offline_client_send = []
        self.online_server_send = [self.blob_masked_input_s]
        self.online_client_send = [self.blob_masked_output_s]


class SwapToClientOfflineServer(SwapToClientOfflineCommon, PhaseProtocolServer):
    input_mask_s = None
    output_s = None
    def __init__(self, num_elem, modulus, name: str):
        super().__init__(num_elem, modulus, name, Config.server_rank)

    def offline(self):
        PhaseProtocolServer.offline(self)
        self.input_mask_s = generate_random_mask(self.modulus, self.num_elem).to(self.comp_device)

    def online(self, input_s):
        input_s = input_s.to(self.comp_device)
        masked_input_s = pmod(input_s + self.input_mask_s, self.modulus)
        self.blob_masked_input_s.send(masked_input_s)

        masked_output_s = self.blob_masked_output_s.get_recv()
        self.output_s = pmod(masked_output_s - self.input_mask_s, self.modulus)


class SwapToClientOfflineClient(SwapToClientOfflineCommon, PhaseProtocolClient):
    output_c = None
    def __init__(self, num_elem, modulus, name: str):
        super().__init__(num_elem, modulus, name, Config.client_rank)

    def offline(self, output_c):
        PhaseProtocolClient.offline(self)
        self.output_c = output_c.to(self.comp_device)

    def online(self, input_c):
        masked_input_s = self.blob_masked_input_s.get_recv()
        input_c = input_c.to(self.comp_device)
        masked_output_s = pmod(masked_input_s + input_c - self.output_c, self.modulus)
        self.blob_masked_output_s.send(masked_output_s)


def test_swap_to_client_comm():
    test_name = "test_swap_to_client_comm"
    print(f"\nTest for {test_name}: Start")
    modulus = 786433
    num_elem = 2 ** 17

    print(f"Number of element: {num_elem}")

    x_s = gen_unirand_int_grain(0, modulus - 1, num_elem).cpu()
    x_c = gen_unirand_int_grain(0, modulus - 1, num_elem).cpu()
    y_c = gen_unirand_int_grain(0, modulus - 1, num_elem).cpu()

    def check_correctness_online(input_s, input_c, output_s, output_c):
        expected = pmod(input_s.cuda() + input_c.cuda(), modulus)
        actual = pmod(output_s.cuda() + output_c.cuda(), modulus)
        compare_expected_actual(expected, actual, name=test_name + " online", get_relative=True)

    def test_server():
        init_communicate(Config.server_rank)
        warming_up_cuda()
        prot = SwapToClientOfflineServer(num_elem, modulus, test_name)
        with NamedTimerInstance("Server Offline"):
            prot.offline()
            torch_sync()
        with NamedTimerInstance("Server online"):
            prot.online(x_s)
            torch_sync()

        blob_output_c = BlobTorch(num_elem, torch.float, prot.comm_base, "output_c")
        blob_output_c.prepare_recv()
        torch_sync()
        output_c = blob_output_c.get_recv()
        check_correctness_online(x_s, x_c, prot.output_s, output_c)

        end_communicate()

    def test_client():
        init_communicate(Config.client_rank)
        warming_up_cuda()
        prot = SwapToClientOfflineClient(num_elem, modulus, test_name)
        with NamedTimerInstance("Client Offline"):
            prot.offline(y_c)
            torch_sync()
        with NamedTimerInstance("Client Online"):
            prot.online(x_c)
            torch_sync()

        blob_output_c = BlobTorch(num_elem, torch.float, prot.comm_base, "output_c")
        torch_sync()
        blob_output_c.send(prot.output_c)
        end_communicate()

    marshal_funcs([test_server, test_client])
    print(f"\nTest for {test_name}: End")


if __name__ == "__main__":
    test_swap_to_client_comm()

