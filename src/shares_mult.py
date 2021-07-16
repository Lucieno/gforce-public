import cProfile, pstats
import time

import torch

from comm import CommFheBuilder, CommBase, PhaseProtocolServer, PhaseProtocolClient, BlobFheEnc, BlobTorch, \
    end_communicate, torch_sync, init_communicate
from config import Config
from fhe import FheBuilder
from timer_utils import NamedTimerInstance
from torch_utils import gen_unirand_int_grain, pmod, marshal_funcs, compare_expected_actual


class SharesMultBase(object):
    def __init__(self, num_elem, modulus, fhe_builder: FheBuilder, name: str, rank):
        self.num_elem = num_elem
        self.modulus = modulus
        self.name = name
        self.fhe_builder = fhe_builder
        self.rank = rank
        self.comm_base = CommBase(rank, name)
        self.comm_fhe = CommFheBuilder(rank, fhe_builder, self.sub_name("comm_fhe"))

        assert(self.modulus == self.fhe_builder.modulus)

        self.blob_fhe_u = BlobFheEnc(num_elem, self.comm_fhe, self.sub_name("fhe_u"))
        self.blob_fhe_v = BlobFheEnc(num_elem, self.comm_fhe, self.sub_name("fhe_v"))
        self.blob_fhe_z_c = BlobFheEnc(num_elem, self.comm_fhe, self.sub_name("fhe_z_c"), ee_mult_time=1)

        self.blob_e_s = BlobTorch(num_elem, torch.float, self.comm_base, "e_s")
        self.blob_f_s = BlobTorch(num_elem, torch.float, self.comm_base, "f_s")
        self.blob_e_c = BlobTorch(num_elem, torch.float, self.comm_base, "e_c")
        self.blob_f_c = BlobTorch(num_elem, torch.float, self.comm_base, "f_c")

        self.offline_server_send = [self.blob_fhe_z_c]
        self.offline_client_send = [self.blob_fhe_u, self.blob_fhe_v]
        self.online_server_send = [self.blob_e_s, self.blob_f_s]
        self.online_client_send = [self.blob_e_c, self.blob_f_c]

    def generate_random(self):
        return gen_unirand_int_grain(0, self.modulus - 1, self.num_elem)

    def mod_to_modulus(self, input):
        return pmod(input, self.modulus)

    def sub_name(self, sub_name: str) -> str:
        return self.name + '_SharesMult_' + sub_name


class SharesMultServer(SharesMultBase, PhaseProtocolServer):
    def __init__(self, num_elem, modulus, fhe_builder: FheBuilder, name: str):
        super().__init__(num_elem, modulus, fhe_builder, name, Config.server_rank)

    def offline(self):
        PhaseProtocolServer.offline(self)
        self.u_s = self.generate_random().to(Config.device)
        self.v_s = self.generate_random().to(Config.device)
        self.z_s = self.generate_random().to(Config.device)

        fhe_u = self.blob_fhe_u.get_recv()
        fhe_u += self.fhe_builder.build_plain_from_torch(self.u_s)
        fhe_v = self.blob_fhe_v.get_recv()
        fhe_v += self.fhe_builder.build_plain_from_torch(self.v_s)
        fhe_z = fhe_u
        fhe_z *= fhe_v
        fhe_z_c = fhe_z
        fhe_z_c -= self.fhe_builder.build_plain_from_torch(self.z_s)
        self.blob_fhe_z_c.send(fhe_z_c)
        del fhe_u, fhe_v, fhe_z, fhe_z_c

    def online(self, a_s, b_s):
        a_s = a_s.to(Config.device)
        b_s = b_s.to(Config.device)
        e_s = self.mod_to_modulus(a_s - self.u_s)
        f_s = self.mod_to_modulus(b_s - self.v_s)
        torch_sync()
        torch_sync()
        self.blob_e_s.send(e_s)
        self.blob_f_s.send(f_s)
        e_s = e_s.to(Config.device).double()
        f_s = f_s.to(Config.device).double()

        e_c = self.blob_e_c.get_recv()
        e = self.mod_to_modulus(e_s + e_c).to(Config.device).double()
        f_c = self.blob_f_c.get_recv()
        f = self.mod_to_modulus(f_s + f_c).to(Config.device).double()
        self.c_s = pmod(a_s * f + e * b_s + self.z_s - e * f, self.modulus)


class SharesMultClient(SharesMultBase, PhaseProtocolClient):
    def __init__(self, num_elem, modulus, fhe_builder: FheBuilder, name: str):
        super().__init__(num_elem, modulus, fhe_builder, name, Config.client_rank)

    def offline(self):
        PhaseProtocolClient.offline(self)
        self.u_c = self.generate_random().double().to(Config.device)
        self.v_c = self.generate_random().double().to(Config.device)
        self.blob_fhe_u.send_from_torch(self.u_c)
        self.blob_fhe_v.send_from_torch(self.v_c)

        self.z_c = self.blob_fhe_z_c.get_recv_decrypt()

    def online(self, a_c, b_c):
        a_c = a_c.to(Config.device)
        b_c = b_c.to(Config.device)
        e_c = self.mod_to_modulus(a_c - self.u_c)
        f_c = self.mod_to_modulus(b_c - self.v_c)
        torch_sync()
        self.blob_e_c.send(e_c)
        self.blob_f_c.send(f_c)
        torch_sync()
        e_c = e_c.to(Config.device).double()
        f_c = f_c.to(Config.device).double()

        e_s = self.blob_e_s.get_recv()
        e = self.mod_to_modulus(e_s + e_c).double().to(Config.device)
        f_s = self.blob_f_s.get_recv()
        f = self.mod_to_modulus(f_s + f_c).double().to(Config.device)
        self.c_c = pmod(a_c * f + e * b_c + self.z_c, self.modulus)


def test_shares_mult():
    print("\nTest for Shares Mult: Start")
    modulus = Config.q_23
    num_elem = 2 ** 17
    print(f"Number of element: {num_elem}")

    fhe_builder = FheBuilder(modulus, Config.n_23)
    fhe_builder.generate_keys()

    a = gen_unirand_int_grain(0, modulus - 1, num_elem)
    a_s = gen_unirand_int_grain(0, modulus - 1, num_elem)
    a_c = pmod(a - a_s, modulus)
    b = gen_unirand_int_grain(0, modulus - 1, num_elem)
    b_s = gen_unirand_int_grain(0, modulus - 1, num_elem)
    b_c = pmod(b - b_s, modulus)

    def check_correctness_offline(u, v, z_s, z_c):
        expected = pmod(u.double().to(Config.device) * v.double().to(Config.device), modulus)
        actual = pmod(z_s + z_c, modulus)
        compare_expected_actual(expected, actual, name="shares_mult_offline", get_relative=True)

    def check_correctness_online(a, b, c_s, c_c):
        expected = pmod(a.double().to(Config.device) * b.double().to(Config.device), modulus)
        actual = pmod(c_s + c_c, modulus)
        compare_expected_actual(expected, actual, name="shares_mult_online", get_relative=True)

    def test_server():
        init_communicate(Config.server_rank)
        prot = SharesMultServer(num_elem, modulus, fhe_builder, "test_shares_mult")
        with NamedTimerInstance("Server Offline"):
            prot.offline()
            torch_sync()
        with NamedTimerInstance("Server Online"):
            prot.online(a_s, b_s)
            torch_sync()

        blob_u_c = BlobTorch(num_elem, torch.float, prot.comm_base, "recon_u_c")
        blob_v_c = BlobTorch(num_elem, torch.float, prot.comm_base, "recon_v_c")
        blob_z_c = BlobTorch(num_elem, torch.float, prot.comm_base, "recon_z_c")
        blob_u_c.prepare_recv()
        blob_v_c.prepare_recv()
        blob_z_c.prepare_recv()
        torch_sync()
        u_c = blob_u_c.get_recv()
        v_c = blob_v_c.get_recv()
        z_c = blob_z_c.get_recv()
        u = pmod(prot.u_s + u_c, modulus)
        v = pmod(prot.v_s + v_c, modulus)
        check_correctness_online(u, v, prot.z_s, z_c)

        blob_c_c = BlobTorch(num_elem, torch.float, prot.comm_base, "c_c")
        blob_c_c.prepare_recv()
        torch_sync()
        c_c = blob_c_c.get_recv()
        check_correctness_online(a, b, prot.c_s, c_c)
        end_communicate()

    def test_client():
        init_communicate(Config.client_rank)
        prot = SharesMultClient(num_elem, modulus, fhe_builder, "test_shares_mult")
        with NamedTimerInstance("Client Offline"):
            prot.offline()
            torch_sync()
        with NamedTimerInstance("Client Online"):
            prot.online(a_c, b_c)
            torch_sync()

        blob_u_c = BlobTorch(num_elem, torch.float, prot.comm_base, "recon_u_c")
        blob_v_c = BlobTorch(num_elem, torch.float, prot.comm_base, "recon_v_c")
        blob_z_c = BlobTorch(num_elem, torch.float, prot.comm_base, "recon_z_c")
        torch_sync()
        blob_u_c.send(prot.u_c)
        blob_v_c.send(prot.v_c)
        blob_z_c.send(prot.z_c)
        blob_c_c = BlobTorch(num_elem, torch.float, prot.comm_base, "c_c")
        torch_sync()
        blob_c_c.send(prot.c_c)
        end_communicate()

    marshal_funcs([test_server, test_client])
    print("\nTest for Shares Mult: End")

if __name__ == "__main__":
    test_shares_mult()
