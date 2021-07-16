import torch

from comm import CommBase, CommFheBuilder, NamedBase, BlobTorch, PhaseProtocolServer, PhaseProtocolClient, \
    init_communicate, torch_sync, end_communicate
from config import Config
from fhe import FheBuilder
from timer_utils import NamedTimerInstance
from torch_utils import generate_random_mask, pmod, gen_unirand_int_grain, compare_expected_actual, marshal_funcs, \
    warming_up_cuda


class ReconToClientCommon(NamedBase):
    class_name = "SwapToClient"
    def __init__(self, num_elem, modulus, name: str, rank):
        super().__init__(name)
        self.rank = rank
        self.name = name
        self.comm_base = CommBase(rank, self.sub_name("comm_base"))
        self.num_elem = num_elem
        self.modulus = modulus
        self.comp_device = torch.device("cpu")

        self.blob_input_s = BlobTorch(self.num_elem, torch.float, self.comm_base, "input_s", dst_device=self.comp_device)

        self.offline_server_send = []
        self.offline_client_send = []
        self.online_server_send = [self.blob_input_s]
        self.online_client_send = []


class ReconToClientServer(ReconToClientCommon, PhaseProtocolServer):
    def __init__(self, num_elem, modulus, name: str):
        super().__init__(num_elem, modulus, name, Config.server_rank)

    def offline(self):
        PhaseProtocolServer.offline(self)

    def online(self, input_s):
        self.blob_input_s.send(input_s)


class ReconToClientClient(ReconToClientCommon, PhaseProtocolClient):
    output = None
    def __init__(self, num_elem, modulus, name: str):
        super().__init__(num_elem, modulus, name, Config.client_rank)

    def offline(self):
        PhaseProtocolClient.offline(self)

    def online(self, input_c):
        input_c = input_c.to(self.comp_device)
        input_s = self.blob_input_s.get_recv()
        self.output = pmod(input_c + input_s, self.modulus)


def test_recon_to_client_comm():
    test_name = "test_recon_to_client_comm"
    print(f"\nTest for {test_name}: Start")
    modulus = 786433
    num_elem = 2 ** 17

    print(f"Number of element: {num_elem}")

    x_s = gen_unirand_int_grain(0, modulus - 1, num_elem).cpu()
    x_c = gen_unirand_int_grain(0, modulus - 1, num_elem).cpu()

    def check_correctness_online(output, input_s, input_c):
        expected = pmod(output.cuda(), modulus)
        actual = pmod(input_s.cuda() + input_c.cuda(), modulus)
        compare_expected_actual(expected, actual, name=test_name + " online", get_relative=True)

    def test_server():
        init_communicate(Config.server_rank)
        warming_up_cuda()
        prot = ReconToClientServer(num_elem, modulus, test_name)
        with NamedTimerInstance("Server Offline"):
            prot.offline()
            torch_sync()
        with NamedTimerInstance("Server online"):
            prot.online(x_s)
            torch_sync()

        end_communicate()

    def test_client():
        init_communicate(Config.client_rank)
        warming_up_cuda()
        prot = ReconToClientClient(num_elem, modulus, test_name)
        with NamedTimerInstance("Client Offline"):
            prot.offline()
            torch_sync()
        with NamedTimerInstance("Client Online"):
            prot.online(x_c)
            torch_sync()

        check_correctness_online(prot.output, x_s, x_c)
        end_communicate()

    marshal_funcs([test_server, test_client])
    print(f"\nTest for {test_name}: End")


if __name__ == "__main__":
    test_recon_to_client_comm()

