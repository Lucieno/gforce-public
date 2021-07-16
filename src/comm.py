import hashlib
import os

import numpy as np
import torch
import torch.distributed as dist
import warnings

from config import Config
from dgk_single_thread import NetworkSimulator
from fhe import FheBuilder, FheEncTensor, encrypt_zeros
from timer_utils import NamedTimerInstance
from torch_utils import compare_expected_actual, warming_up_cuda, torch_from_buffer, marshal_funcs, get_prod, \
    generate_random_mask, get_num_byte


def str_hash(s):
    return int(int(hashlib.sha224(s.encode('utf-8')).hexdigest(), 16) % ((1 << 62) - 1))


def init_communicate(rank, master_address="127.0.0.1", master_port="12595", backend='gloo'):
    os.environ['MASTER_ADDR'] = master_address
    os.environ['MASTER_PORT'] = master_port
    print(f"rank: {rank}, master_address: {master_address}, master_port: {master_port}")
    dist.init_process_group(backend, rank=rank, world_size=Config.world_size)

def end_communicate():
    dist.destroy_process_group()

def torch_sync():
    dist.barrier()


class NamedBase(object):
    class_name = "NameBase"
    def __init__(self, name):
        self.name = name

    def sub_name(self, sub_name: str) -> str:
        return self.name + '_' + self.class_name + '_' + sub_name


class TrafficRecord(object):
    class __TrafficRecord(object):
        def __init__(self):
            self.num_sent_byte = 0
            self.num_recv_byte = 0
            self.sent_record = {}
            self.recv_record = {}

        def reset(self, name, verbose=True):
            if verbose:
                total = self.num_sent_byte + self.num_recv_byte
                print(f"{name} sent byte: {self.num_sent_byte}, recv byte: {self.num_recv_byte}, total: {total}")
            self.sent_record[name] = self.num_sent_byte
            self.recv_record[name] = self.num_recv_byte
            self.num_sent_byte = 0
            self.num_recv_byte = 0

        def send_byte(self, num_byte):
            self.num_sent_byte += num_byte

        def recv_byte(self, num_byte):
            self.num_recv_byte += num_byte

    instance = None

    def __new__(cls): # __new__ always a classmethod
        if not TrafficRecord.instance:
            TrafficRecord.instance = TrafficRecord.__TrafficRecord()
        return TrafficRecord.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)


class CommBase(object):
    def __init__(self, rank: int, name: str):
        self.name = name
        self.rank = rank
        self.dst = 1 - rank
        self.all_wait = dict()
        self.all_tensors = dict()
        self.traffic_record = TrafficRecord()

    def get_dist_tag(self, name: str) -> str:
        return str_hash(str(self.name) + "_for_" + str(name)) % ((1 << 31) - 1)

    def send_torch(self, torch_tensor: torch.Tensor, name: str):
        tag = self.get_dist_tag(name)
        send_tensor = torch_tensor.cpu()
        self.all_wait[tag] = dist.isend(tensor=send_tensor, dst=self.dst, tag=tag)

        self.traffic_record.send_byte(get_num_byte(send_tensor))

    def recv_torch(self, torch_tensor: torch.Tensor, name: str):
        tag = self.get_dist_tag(name)
        self.all_tensors[tag] = torch_tensor
        self.all_wait[tag] = dist.irecv(tensor=self.all_tensors[tag], src=self.dst, tag=tag)


    def wait(self, name: str):
        tag = self.get_dist_tag(name)
        if tag not in self.all_wait:
            raise Exception("Tensor %s and %d has not been isend/irecv" % (name, self.dst))
        self.all_wait[tag].wait()
        self.traffic_record.recv_byte(get_num_byte(self.all_tensors[tag]))
        del self.all_wait[tag]

    def get_tensor(self, name: str) -> torch.Tensor:
        tag = self.get_dist_tag(name)
        return self.all_tensors[tag]


class CommFheBuilder(object):
    def __init__(self, rank: int, fhe_builder: FheBuilder, name: str):
        self.comm_base = CommBase(rank, name)
        self.fhe_builder = fhe_builder
        self.recv_list = dict()

    def sub_name(self, name: str, i: int):
        return name+'_'+str(i)

    def send_public_key(self):
        return self.comm_base.send_torch(self.fhe_builder.get_public_key_buffer(), "public_key")

    def recv_public_key(self):
        return self.comm_base.recv_torch(self.fhe_builder.get_public_key_buffer(), "public_key")

    def wait_and_build_public_key(self):
        self.comm_base.wait("public_key")
        self.fhe_builder.build_from_loaded_public_key()

    def send_secret_key(self):
        return self.comm_base.send_torch(self.fhe_builder.get_secret_key_buffer(), "secret_key")

    def recv_secret_key(self):
        return self.comm_base.recv_torch(self.fhe_builder.get_secret_key_buffer(), "secret_key")

    def wait_and_build_secret_key(self):
        self.comm_base.wait("secret_key")
        self.fhe_builder.build_from_loaded_secret_key()

    def send_raw_cts(self, cts, num_batch: int, name: str):
        for i in range(num_batch):
            self.comm_base.send_torch(torch_from_buffer(cts[i]), self.sub_name(name, i))

    def inject_noise_before_send(self, enc: FheEncTensor):
        if self.fhe_builder.degree < Config.safe_degree:
            warnings.warn(f"The degree {self.fhe_builder.degree} is too small for noise injection for circuit privacy. "
                          f"The noise injection is skipped")
            return
        evaluator = self.fhe_builder.evaluator
        for ct in enc.cts:
            evaluator.add_noise(ct, Config.noise_bit)

    def send_enc(self, enc: FheEncTensor, name: str):
        self.inject_noise_before_send(enc)
        self.send_raw_cts(enc.cts, enc.num_batch, name)

    def recv_raw_cts(self, cts, num_batch: int, name: str):
        self.recv_list[name] = list()
        for i in range(num_batch):
            sub_name = self.sub_name(name, i)
            self.comm_base.recv_torch(torch_from_buffer(cts[i]), sub_name)
            self.recv_list[name].append(sub_name)

    def recv_enc(self, enc: FheEncTensor, name: str):
        self.recv_raw_cts(enc.cts, enc.num_batch, name)

    def wait_enc(self, name: str):
        if name not in self.recv_list:
            raise Exception(f"{name} is not in the receiving list")
        for sub_name in self.recv_list[name]:
            self.comm_base.wait(sub_name)


class PhaseProtocolCommon(object):
    offline_client_send: list
    online_client_send: list
    offline_server_send: list
    online_server_send: list
    output_s: torch.Tensor
    output_c: torch.Tensor

    def __init__(self):
        pass

    def offline(self, *args, **kwargs):
        raise NotImplementedError()

    def online(self, *args, **kwargs):
        raise NotImplementedError()


class PhaseProtocolServer(PhaseProtocolCommon):
    def __init__(self):
        super().__init__()

    def offline_recv(self):
        for blob in self.offline_client_send:
            blob.prepare_recv()

    def online_recv(self):
        for blob in self.online_client_send:
            blob.prepare_recv()

    def offline(self, *args, **kwargs):
        self.offline_recv()
        self.online_recv()
        torch_sync()


class PhaseProtocolClient(PhaseProtocolCommon):
    def __init__(self):
        super().__init__()

    def offline_recv(self):
        for blob in self.offline_server_send:
            blob.prepare_recv()

    def online_recv(self):
        for blob in self.online_server_send:
            blob.prepare_recv()

    def offline(self, *args, **kwargs):
        self.offline_recv()
        self.online_recv()
        torch_sync()


class BlobTorch(object):
    recv_tensor = None
    def __init__(self, shape, transfer_dtype, comm: CommBase, name: str, comp_dtype=None, dst_device="cuda:0"):
        if isinstance(shape, torch.Size):
            self.shape = shape
        elif isinstance(shape, list):
            self.shape = torch.Size(shape)
        else:
            self.shape = torch.Size([shape])
        if not Config.use_cuda:
            self.dst_device = "cpu"
        else:
            self.dst_device = dst_device
        self.name = name
        self.comm = comm
        self.transfer_dtype = transfer_dtype
        self.comp_dtype = transfer_dtype if comp_dtype is None else comp_dtype

    def send(self, send_tensor: torch.Tensor):
        assert(send_tensor.shape == self.shape)
        self.comm.send_torch(send_tensor.type(self.transfer_dtype), self.name)

    def prepare_recv(self):
        self.recv_tensor = torch.zeros(self.shape, dtype=self.transfer_dtype)
        self.comm.recv_torch(self.recv_tensor, self.name)

    def get_recv(self):
        self.comm.wait(self.name)
        return self.recv_tensor.to(self.dst_device).type(self.comp_dtype)


class BlobFheRawCts(object):
    recv_tensor = None
    def __init__(self, num_batch: int, comm: CommFheBuilder, name: str):
        self.name = name
        self.comm = comm
        self.num_batch = num_batch
        self.fhe_builder = comm.fhe_builder
        self.batch_encoder = self.fhe_builder.batch_encoder
        self.encryptor = self.fhe_builder.encryptor
        self.degree = self.fhe_builder.degree

    def send(self, cts):
        assert(len(cts) == self.num_batch)
        self.comm.send_raw_cts(cts, self.num_batch, self.name)

    def prepare_recv(self):
        self.recv_tensor = encrypt_zeros(self.num_batch, self.batch_encoder, self.encryptor, self.degree)
        self.comm.recv_raw_cts(self.recv_tensor, self.num_batch, self.name)

    def get_recv(self):
        self.comm.wait_enc(self.name)
        return self.recv_tensor


class BlobFheEnc(object):
    recv_tensor = None
    def __init__(self, num_elem: int, comm: CommFheBuilder, name: str, ee_mult_time=0):
        self.name = name
        self.comm = comm
        self.num_elem = num_elem
        self.fhe_builder = comm.fhe_builder
        self.ee_mult_time = ee_mult_time

    def send(self, enc: FheEncTensor):
        assert(enc.num_elem == self.num_elem)
        self.comm.send_enc(enc, self.name)

    def send_from_torch(self, torch_tensor):
        assert(len(torch_tensor.shape) == 1)
        assert(torch_tensor.shape[0] == self.num_elem)
        enc = self.fhe_builder.build_enc_from_torch(torch_tensor)
        self.send(enc)

    def prepare_for_ee_mult_ciphers(self):
        self.recv_tensor = self.comm.fhe_builder.build_enc(self.num_elem, is_cheap_init=True)
        self.recv_tensor.dummy_ee_multed()

    def prepare_recv(self):
        if self.ee_mult_time == 0:
            self.recv_tensor = self.comm.fhe_builder.build_enc(self.num_elem)
        elif self.ee_mult_time == 1:
            self.prepare_for_ee_mult_ciphers()
        else:
            raise NotImplementedError()
        self.comm.recv_enc(self.recv_tensor, self.name)

    def get_recv(self) -> FheEncTensor:
        self.comm.wait_enc(self.name)
        return self.recv_tensor

    def get_recv_decrypt(self):
        enc = self.get_recv()
        return self.fhe_builder.decrypt_to_torch(enc)


class BlobFheEnc2D(object):
    recv_tensor = None
    def __init__(self, shape, comm: CommFheBuilder, name: str):
        assert(len(shape) == 2)
        self.name = name
        self.comm = comm
        self.shape = shape
        self.num_enc = shape[0]
        self.num_elem = shape[1]
        self.fhe_builder = comm.fhe_builder

    def sub_name(self, i: int):
        return self.name + '_' + str(i)

    def send(self, list_enc: list):
        assert(isinstance(list_enc[0], FheEncTensor))
        assert(len(list_enc) == self.num_enc)
        assert(list_enc[0].num_elem == self.num_elem)
        for i in range(len(list_enc)):
            self.comm.send_enc(list_enc[i], self.sub_name(i))

    def send_from_torch(self, torch_tensor):
        assert(torch_tensor.shape == torch.Size(self.shape))
        enc = [self.fhe_builder.build_enc_from_torch(torch_tensor[i]) for i in range(len(torch_tensor))]
        self.send(enc)

    def prepare_recv(self):
        self.recv_tensor = []
        for i in range(self.num_enc):
            self.recv_tensor.append(self.comm.fhe_builder.build_enc(self.num_elem))
            self.comm.recv_enc(self.recv_tensor[i], self.sub_name(i))

    def get_recv(self) -> FheEncTensor:
        for i in range(self.num_enc):
            self.comm.wait_enc(self.sub_name(i))
        return self.recv_tensor

    def get_recv_decrypt(self):
        enc = self.get_recv()
        return self.fhe_builder.decrypt_to_torch(enc)

def test_comm_fhe_builder():
    num_elem = 2 ** 17
    # modulus, degree = 12289, 2048
    modulus, degree = Config.q_16, Config.n_16
    expected_tensor_pk = torch.zeros(num_elem).float() + 23
    expected_tensor_sk = torch.zeros(num_elem).float() + 42
    pk_tag = "pk"
    sk_tag = "sk"
    ori_tag = "ori"

    ori = generate_random_mask(modulus, num_elem)

    def test_server():
        init_communicate(Config.server_rank)
        fhe_builder = FheBuilder(modulus, degree)
        comm_fhe_builder = CommFheBuilder(Config.server_rank, fhe_builder, "fhe_builder")

        comm_fhe_builder.recv_public_key()
        comm_fhe_builder.wait_and_build_public_key()

        blob_ori = BlobFheEnc(num_elem, comm_fhe_builder, ori_tag)
        blob_ori.send_from_torch(ori)

        enc_pk = fhe_builder.build_enc_from_torch(expected_tensor_pk)
        comm_fhe_builder.send_enc(enc_pk, pk_tag)

        comm_fhe_builder.recv_secret_key()
        comm_fhe_builder.wait_and_build_secret_key()

        enc_sk = fhe_builder.build_enc(num_elem)
        comm_fhe_builder.recv_enc(enc_sk, sk_tag)
        comm_fhe_builder.wait_enc(sk_tag)
        actual_tensor_sk = fhe_builder.decrypt_to_torch(enc_sk)
        compare_expected_actual(expected_tensor_sk, actual_tensor_sk, get_relative=True, name="Recovering to test sk")

        dist.destroy_process_group()

    def test_client():
        init_communicate(Config.client_rank)
        fhe_builder = FheBuilder(modulus, degree)
        comm_fhe_builder = CommFheBuilder(Config.client_rank, fhe_builder, "fhe_builder")
        fhe_builder.generate_keys()

        comm_fhe_builder.send_public_key()

        blob_ori = BlobFheEnc(num_elem, comm_fhe_builder, ori_tag)
        blob_ori.prepare_recv()
        dec = blob_ori.get_recv_decrypt()
        compare_expected_actual(ori, dec, get_relative=True, name=ori_tag)

        enc_pk = fhe_builder.build_enc(num_elem)
        comm_fhe_builder.recv_enc(enc_pk, pk_tag)
        comm_fhe_builder.wait_enc(pk_tag)
        actual_tensor_pk = fhe_builder.decrypt_to_torch(enc_pk)
        compare_expected_actual(expected_tensor_pk, actual_tensor_pk, get_relative=True, name="Recovering to test pk")

        comm_fhe_builder.send_secret_key()

        enc_sk = fhe_builder.build_enc_from_torch(expected_tensor_sk)
        comm_fhe_builder.send_enc(enc_sk, sk_tag)

        dist.destroy_process_group()

    marshal_funcs([test_server, test_client])


def test_comm_base():
    num_elem = 2 ** 17
    expect_float = torch.ones(num_elem).float() + 3
    expect_double = torch.ones(num_elem).double() + 5
    expect_int16 = torch.ones(num_elem).type(torch.int16) + 324
    expect_uint8 = torch.ones(num_elem).type(torch.uint8) + 123
    comm_name = "comm_base"
    float_tag = "float_tag"
    double_tag = "double_tag"
    int16_tag = "int16_tag"
    uint8_tag = "uint8_tag"

    def comm_base_server():
        init_communicate(Config.server_rank)
        comm_base = CommBase(Config.server_rank, comm_name)

        send_int16 = expect_int16.cuda()
        send_uint8 = expect_uint8.cuda()

        with NamedTimerInstance("Server float and int16"):
            comm_base.recv_torch(torch.zeros(num_elem).float(), float_tag)
            comm_base.send_torch(send_int16, int16_tag)
            comm_base.wait(float_tag)
            actual_float = comm_base.get_tensor(float_tag).cuda()

        comm_base.recv_torch(torch.zeros(num_elem).float(), float_tag)
        dist.barrier()
        with NamedTimerInstance("Server float and int16"):
            comm_base.send_torch(send_int16, int16_tag)
            comm_base.wait(float_tag)
            actual_float = comm_base.get_tensor(float_tag).cuda()

        comm_base.recv_torch(torch.zeros(num_elem).double(), double_tag)
        dist.barrier()
        with NamedTimerInstance("Server double and uint8"):
            comm_base.send_torch(send_uint8, uint8_tag)
            comm_base.wait(double_tag)
            actual_double = comm_base.get_tensor(double_tag).cuda()

        comm_base.recv_torch(torch.zeros(num_elem).double(), double_tag)
        dist.barrier()
        with NamedTimerInstance("Server double and uint8"):
            comm_base.send_torch(send_uint8, uint8_tag)
            comm_base.wait(double_tag)
            actual_double = comm_base.get_tensor(double_tag).cuda()

        dist.barrier()
        compare_expected_actual(expect_float.cuda(), actual_float, name="float", get_relative=True)
        compare_expected_actual(expect_double.cuda(), actual_double, name="double", get_relative=True)

        google_vm_simulator = NetworkSimulator(bandwidth=10 * (10 ** 9), basis_latency=.001)
        with NamedTimerInstance("Simulate int16"):
            google_vm_simulator.simulate(send_int16.cpu().cuda())
        with NamedTimerInstance("Simulate uint8"):
            google_vm_simulator.simulate(send_uint8.cpu().cuda())

        dist.destroy_process_group()

    def comm_base_client():
        init_communicate(Config.client_rank)
        comm_base = CommBase(Config.client_rank, comm_name)

        send_float = expect_float.cuda()
        send_double = expect_double.cuda()

        with NamedTimerInstance("Client float and int16"):
            comm_base.recv_torch(torch.zeros(num_elem).type(torch.int16), int16_tag)
            comm_base.send_torch(send_float, float_tag)
            comm_base.wait(int16_tag)
            actual_int16 = comm_base.get_tensor(int16_tag).cuda()

        comm_base.recv_torch(torch.zeros(num_elem).type(torch.int16), int16_tag)
        dist.barrier()
        with NamedTimerInstance("Client float and int16"):
            comm_base.send_torch(send_float, float_tag)
            comm_base.wait(int16_tag)
            actual_int16 = comm_base.get_tensor(int16_tag).cuda()

        comm_base.recv_torch(torch.zeros(num_elem).type(torch.uint8), uint8_tag)
        dist.barrier()
        with NamedTimerInstance("Client double and uint8"):
            comm_base.send_torch(send_double, double_tag)
            comm_base.wait(uint8_tag)
            actual_uint8 = comm_base.get_tensor(uint8_tag).cuda()

        comm_base.recv_torch(torch.zeros(num_elem).type(torch.uint8), uint8_tag)
        dist.barrier()
        with NamedTimerInstance("Client double and uint8"):
            comm_base.send_torch(send_double, double_tag)
            comm_base.wait(uint8_tag)
            actual_uint8 = comm_base.get_tensor(uint8_tag).cuda()

        dist.barrier()
        compare_expected_actual(expect_int16.cuda(), actual_int16, name="int16", get_relative=True)
        compare_expected_actual(expect_uint8.cuda(), actual_uint8, name="uint8", get_relative=True)

        dist.destroy_process_group()

    marshal_funcs([comm_base_server, comm_base_client])

if __name__ == "__main__":
    # test_comm_base()
    test_comm_fhe_builder()
