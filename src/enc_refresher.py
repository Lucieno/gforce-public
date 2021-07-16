import torch.distributed as dist

from comm import CommFheBuilder, BlobFheEnc, BlobFheEnc2D, init_communicate, end_communicate, torch_sync
from config import Config
from fhe import sub_handle, FheBuilder
from timer_utils import NamedTimerInstance
from torch_utils import gen_unirand_int_grain, compare_expected_actual, marshal_funcs


class EncRefresherCommon(object):
    def __init__(self, shape, comm: CommFheBuilder, name: str):
        self.shape = shape
        self.comm_fhe_builder = comm
        self.fhe_builder = comm.fhe_builder
        self.modulus = self.fhe_builder.modulus

        if isinstance(shape, int) or len(shape) == 1:
            Blob = BlobFheEnc
        elif len(shape) == 2:
            Blob = BlobFheEnc2D

        self.masked = Blob(shape, self.comm_fhe_builder, f"{name}_masked")
        self.refreshed = Blob(shape, self.comm_fhe_builder, f"{name}_refreshed")


def delete_fhe(enc):
    if isinstance(enc, list):
        for x in enc:
            del x
    del enc


class EncRefresherServer(object):
    r_s = None
    def __init__(self, shape, fhe_builder: FheBuilder, name: str):
        self.name = name
        self.shape = shape
        self.fhe_builder = fhe_builder
        self.modulus = self.fhe_builder.modulus
        comm = CommFheBuilder(Config.server_rank, fhe_builder, name)
        self.common = EncRefresherCommon(shape, comm, name)
        self.is_prepared = False

    def prepare_recv(self):
        if self.is_prepared:
            return
        self.is_prepared = True
        self.common.refreshed.prepare_recv()

    def request(self, enc):
        self.prepare_recv()
        torch_sync()
        self.r_s = gen_unirand_int_grain(0, self.modulus - 1, self.shape)

        if len(self.shape) == 2:
            pt = []
            for i in range(self.shape[0]):
                pt.append(self.fhe_builder.build_plain_from_torch(self.r_s[i]))
                enc[i] += pt[i]

            self.common.masked.send(enc)
            refreshed = self.common.refreshed.get_recv()

            for i in range(self.shape[0]):
                refreshed[i] -= pt[i]
            delete_fhe(enc)
            delete_fhe(pt)
            torch_sync()
            return refreshed
        else:
            pt = self.fhe_builder.build_plain_from_torch(self.r_s)
            enc += pt
            self.common.masked.send(enc)
            refreshed = self.common.refreshed.get_recv()
            refreshed -= pt
            delete_fhe(enc)
            delete_fhe(pt)
            torch_sync()
            return refreshed

        # def sub_request(sub_enc, sub_r_s):
        #     pt = self.fhe_builder.build_plain_from_torch(sub_r_s)
        #     sub_enc += pt
        #     del pt
        #     return sub_enc
        # fhe_masked = sub_handle(sub_request, enc, self.r_s)
        # with NamedTimerInstance("Server refresh round"):
        #     self.common.masked.send(fhe_masked)
        #
        #     refreshed = self.common.refreshed.get_recv()
        # def sub_unmask(sub_enc, sub_r_s):
        #     pt = self.fhe_builder.build_plain_from_torch(sub_r_s)
        #     sub_enc -= pt
        #     del pt
        #     return sub_enc
        # unmasked_refreshed = sub_handle(sub_unmask, refreshed, self.r_s)

        # torch_sync()
        # delete_fhe(enc)
        # delete_fhe(fhe_masked)
        # delete_fhe(refreshed)
        #
        # return unmasked_refreshed


class EncRefresherClient(object):
    r_s = None
    def __init__(self, shape, fhe_builder: FheBuilder, name: str):
        self.shape = shape
        self.fhe_builder = fhe_builder
        self.modulus = self.fhe_builder.modulus
        comm = CommFheBuilder(Config.client_rank, fhe_builder, name)
        self.common = EncRefresherCommon(shape, comm, name)
        self.is_prepared = False

    def prepare_recv(self):
        if self.is_prepared:
            return
        self.is_prepared = True
        self.common.masked.prepare_recv()

        if len(self.shape) == 2:
            self.refreshed = []
            for i in range(self.shape[0]):
                self.refreshed.append(self.fhe_builder.build_enc(self.shape[1]))
        else:
            self.refreshed = self.fhe_builder.build_enc(self.shape[0])

    def response(self):
        self.prepare_recv()
        torch_sync()

        fhe_masked = self.common.masked.get_recv()

        with NamedTimerInstance("client refresh reencrypt"):
            if len(self.shape) == 2:
                for i in range(self.shape[0]):
                    sub_dec = self.fhe_builder.decrypt_to_torch(fhe_masked[i])
                    self.refreshed[i].encrypt_additive(sub_dec)
            else:
                self.refreshed.encrypt_additive(self.fhe_builder.decrypt_to_torch(fhe_masked))

        # def sub_reencrypt(sub_enc):
        #     sub_dec = self.fhe_builder.decrypt_to_torch(sub_enc)
        #     sub_refreshed = self.fhe_builder.build_enc_from_torch(sub_dec)
        #     del sub_dec
        #     return sub_refreshed
        # with NamedTimerInstance("client refresh reencrypt"):
        #     self.refreshed = sub_handle(sub_reencrypt, fhe_masked)

        self.common.refreshed.send(self.refreshed)

        torch_sync()
        delete_fhe(fhe_masked)
        delete_fhe(self.refreshed)


def test_fhe_refresh():
    print()
    print("Test: FHE refresh: Start")
    modulus, degree = 12289, 2048
    num_batch = 12
    num_elem = 2 ** 15
    fhe_builder = FheBuilder(modulus, degree)
    fhe_builder.generate_keys()

    def test_batch_server():
        init_communicate(Config.server_rank)
        shape = [num_batch, num_elem]
        tensor = gen_unirand_int_grain(0, modulus, shape)
        refresher = EncRefresherServer(shape, fhe_builder, "test_batch_refresher")
        enc = [fhe_builder.build_enc_from_torch(tensor[i]) for i in range(num_batch)]
        refreshed = refresher.request(enc)
        tensor_refreshed = fhe_builder.decrypt_to_torch(refreshed)
        compare_expected_actual(tensor, tensor_refreshed, get_relative=True, name="batch refresh")

        end_communicate()

    def test_batch_client():
        init_communicate(Config.client_rank)
        shape = [num_batch, num_elem]
        refresher = EncRefresherClient(shape, fhe_builder, "test_batch_refresher")
        refresher.prepare_recv()
        refresher.response()

        end_communicate()

    marshal_funcs([test_batch_server, test_batch_client])

    def test_1d_server():
        init_communicate(Config.server_rank)
        shape = num_elem
        tensor = gen_unirand_int_grain(0, modulus, shape)
        refresher = EncRefresherServer(shape, fhe_builder, "test_1d_refresher")
        enc = fhe_builder.build_enc_from_torch(tensor)
        refreshed = refresher.request(enc)
        tensor_refreshed = fhe_builder.decrypt_to_torch(refreshed)
        compare_expected_actual(tensor, tensor_refreshed, get_relative=True, name="1d_refresh")

        end_communicate()

    def test_1d_client():
        init_communicate(Config.client_rank)
        shape = num_elem
        refresher = EncRefresherClient(shape, fhe_builder, "test_1d_refresher")
        refresher.prepare_recv()
        refresher.response()

        end_communicate()

    marshal_funcs([test_1d_server, test_1d_client])

    print()
    print("Test: FHE refresh: End")

if __name__ == "__main__":
    test_fhe_refresh()
