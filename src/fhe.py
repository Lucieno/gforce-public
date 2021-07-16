import numpy as np

import torch

from config import Config
from timer_utils import NamedTimerInstance
from torch_utils import get_prod, compare_expected_actual, gen_unirand_int_grain, pmod, torch_from_buffer
from seal import scheme_type, EncryptionParameters, CoeffModulus, SEALContext, KeyGenerator, Encryptor, Decryptor, \
    BatchEncoder, Evaluator, Plaintext, uIntVector, Ciphertext


class FheTensor(object):
    def __init__(self, num_elem, modulus, degree, context=None, evaluator=None, batch_encoder=None, encryptor=None):
        self.num_elem = num_elem
        self.modulus = modulus
        self.degree = degree
        self.context = context
        self.parms = context.first_context_data().parms()
        self.evaluator = evaluator
        self.batch_encoder = batch_encoder
        self.encryptor = encryptor
        self.acceptable_degree = [1024, 2048, 4096, 8192, 16384]
        if self.degree not in self.acceptable_degree:
            raise Exception(f"Get {self.degree}, but expect degree {self.acceptable_degree}")
        self.num_batch = (((self.num_elem - 1) // self.degree) + 1)
        self.num_incomplete_batch = 0 if self.num_elem % self.degree == 0 else 1
        self.num_full_batch = self.num_batch - self.num_incomplete_batch
        self.padded_num_slot = self.num_batch * self.degree
        self.num_last_elem = self.num_elem - self.num_full_batch * self.degree

        assert(self.parms.poly_modulus_degree() == degree)

    def dim_check(self, other):
        if self.num_elem != other.num_elem:
            raise Exception(f"num_elem should be the same: self: {self.num_elem}, other: {other.num_elem}")
        if self.modulus != other.modulus:
            raise Exception(f"modulus should be the same: self: {self.modulus}, other: {other.modulus}")
        if self.degree != other.degree:
            raise Exception(f"degree should be the same: self: {self.degree}, other: {other.degree}")


class FhePlainTensor(FheTensor):
    def __init__(self, num_elem, modulus, degree, context, evaluator, batch_encoder):
        super().__init__(num_elem, modulus, degree, context=context, evaluator=evaluator, batch_encoder=batch_encoder)
        self.pod_vector = uIntVector()
        self.batched_pts = [Plaintext(self.degree, 0) for _ in range(self.num_batch)]

    def load_from_torch(self, torch_tensor):
        torch_tensor = torch_tensor.reshape(-1)
        assert(torch_tensor.shape[0] == self.num_elem)

        transformed_tensor = torch_tensor.cpu().numpy().astype(np.uint64)

        for i in range(self.num_full_batch):
            self.pod_vector.from_np(transformed_tensor[self.degree*i:self.degree*(i+1)])
            self.batch_encoder.encode(self.pod_vector, self.batched_pts[i])

        if self.num_incomplete_batch > 0:
            padded = np.zeros(self.degree).astype(np.uint64)
            padded[:self.num_last_elem] = transformed_tensor[-self.num_last_elem:]
            self.pod_vector.from_np(padded)
            self.batch_encoder.encode(self.pod_vector, self.batched_pts[-1])

    def export_as_torch(self, to="gpu"):
        device = torch.device("cuda:0") if to == "gpu" else torch.device("cpu")
        res = torch.zeros(self.num_elem, device=device).type(torch.float)

        for i in range(self.num_full_batch):
            self.batch_encoder.decode(self.batched_pts[i], self.pod_vector)
            arr = np.array(self.pod_vector, copy=False)
            res[self.degree*i:self.degree*(i+1)].copy_(torch.from_numpy(arr.astype(np.float)))

        if self.num_incomplete_batch > 0:
            self.batch_encoder.decode(self.batched_pts[-1], self.pod_vector)
            arr = np.array(self.pod_vector, copy=False)
            res[-self.num_last_elem:].copy_(torch.from_numpy(arr[:self.num_last_elem].astype(np.float)))

        return res


def encrypt_zeros(num_batch, batch_encoder, encryptor, degree):
    cts = [Ciphertext() for i in range(num_batch)]
    pod_vector = uIntVector()
    pt = Plaintext()
    zeros_tensor = np.zeros(degree).astype(np.int64)
    pod_vector.from_np(zeros_tensor)
    batch_encoder.encode(pod_vector, pt)
    for i in range(num_batch):
        encryptor.encrypt(pt, cts[i])
    return cts


class FheEncTensor(FheTensor):
    def __init__(self, num_elem, modulus, degree, context, evaluator, batch_encoder, encryptor, is_cheap_init=False):
        super().__init__(num_elem, modulus, degree,
                         context=context, evaluator=evaluator, batch_encoder=batch_encoder, encryptor=encryptor)
        self.pod_vector = uIntVector()
        self.pt = Plaintext()
        self.cts = [Ciphertext() for i in range(self.num_batch)]
        self.multiply_enc_quota = 1
        self.multiply_enc_times = 0
        self.multiply_plain_quota = 1
        self.multiply_plain_times = 0

        if not is_cheap_init:
            self.encrypt_zeros()
            self.all_zeros = True

    def copy(self):
        res = FheEncTensor(self.num_elem, self.modulus, self.degree, self.context, self.evaluator,
                           self.batch_encoder, self.encryptor, is_cheap_init=True)
        for i in range(self.num_batch):
            res.cts[i] = Ciphertext(self.cts[i])
        res.multiply_enc_quota = self.multiply_enc_quota
        res.multiply_enc_times = self.multiply_enc_times
        res.multiply_plain_quota = self.multiply_plain_quota
        res.multiply_plain_times = self.multiply_plain_times
        res.all_zeros = self.all_zeros
        return res

    def dummy_ee_multed(self):
        zeros_tensor = np.zeros(self.degree).astype(np.int64)
        self.pod_vector.from_np(zeros_tensor)
        self.batch_encoder.encode(self.pod_vector, self.pt)
        self.encryptor.encrypt(self.pt, self.cts[0])
        self.evaluator.multiply_inplace(self.cts[0], self.cts[0])
        for i in range(1, self.num_batch):
            self.cts[i] = Ciphertext(self.cts[0])
        self.multiply_enc_times = 1

    def encrypt_zeros(self):
        self.cts = encrypt_zeros(self.num_batch, self.batch_encoder, self.encryptor, self.degree)

    def encrypt_additive(self, torch_tensor):
        torch_tensor = torch_tensor.reshape(-1)
        # assert(len(torch_tensor.shape) == 1)
        assert(torch_tensor.shape[0] == self.num_elem)
        assert(self.all_zeros is True)

        transformed_tensor = torch_tensor.cpu().numpy().astype(np.uint64)

        for i in range(self.num_full_batch):
            self.pod_vector.from_np(transformed_tensor[self.degree*i:self.degree*(i+1)])
            self.batch_encoder.encode(self.pod_vector, self.pt)
            self.evaluator.add_plain_inplace(self.cts[i], self.pt)

        if self.num_incomplete_batch > 0:
            padded_arr = np.zeros(self.degree, np.uint64)
            padded_arr[:self.num_last_elem] = transformed_tensor[-self.num_last_elem:]
            self.pod_vector.from_np(padded_arr)
            self.batch_encoder.encode(self.pod_vector, self.pt)
            self.evaluator.add_plain_inplace(self.cts[-1], self.pt)

        self.all_zeros = False

    def decrypt(self, decryptor, dst=None, to="gpu"):
        if dst is None:
            device = torch.device(Config.device_name) if to == "gpu" else torch.device("cpu")
            dst = torch.zeros(self.num_elem, device=device).type(torch.float)
        else:
            assert(len(dst.shape) == 1)
            assert(dst.shape[0] == self.num_elem)

        for i in range(self.num_full_batch):
            decryptor.decrypt(self.cts[i], self.pt)
            self.batch_encoder.decode(self.pt, self.pod_vector)
            arr = np.array(self.pod_vector, copy=False)
            dst[self.degree * i:self.degree * (i + 1)].copy_(torch.from_numpy(arr.astype(np.float)))

        if self.num_incomplete_batch > 0:
            decryptor.decrypt(self.cts[-1], self.pt)
            self.batch_encoder.decode(self.pt, self.pod_vector)
            arr = np.array(self.pod_vector, copy=False)
            dst[-self.num_last_elem:].copy_(torch.from_numpy(arr.astype(np.float)[:self.num_last_elem]))

        return dst

    def ep_mul_unsafe(self, other: FhePlainTensor):
        for i in range(self.num_batch):
            self.evaluator.multiply_plain_inplace(self.cts[i], other.batched_pts[i])

    def ee_mul_unsafe(self, other: 'FheEncTensor'):
        for i in range(self.num_batch):
            self.evaluator.multiply_inplace(self.cts[i], other.cts[i])

    def ep_add_unsafe(self, other: FhePlainTensor):
        for i in range(self.num_batch):
            self.evaluator.add_plain_inplace(self.cts[i], other.batched_pts[i])

    def ee_add_unsafe(self, other: 'FheEncTensor'):
        for i in range(self.num_batch):
            self.evaluator.add_inplace(self.cts[i], other.cts[i])

    def ep_sub_unsafe(self, other: FhePlainTensor):
        for i in range(self.num_batch):
            self.evaluator.sub_plain_inplace(self.cts[i], other.batched_pts[i])

    def ee_sub_unsafe(self, other: 'FheEncTensor'):
        for i in range(self.num_batch):
            self.evaluator.sub_inplace(self.cts[i], other.cts[i])

    def __imul__(self, other):
        self.dim_check(other)
        if isinstance(other, FhePlainTensor):
            if self.multiply_plain_times >= self.multiply_plain_quota:
                raise Exception(f"Too many plain-multiplication. "
                                f"Quota: {self.multiply_plain_quota}. Times-ed: {self.multiply_plain_times}")
            self.ep_mul_unsafe(other)
            self.multiply_plain_times += 1
            return self

        if isinstance(other, FheEncTensor):
            if self.multiply_enc_times >= self.multiply_enc_quota:
                raise Exception(f"Too many plain-multiplication. "
                                f"Quota: {self.multiply_enc_quota}. Times-ed: {self.multiply_enc_times}")
            self.ee_mul_unsafe(other)
            self.multiply_enc_quota += 1
            return self

        raise Exception(f"Unknown rvalue type: {type(other)}")

    def __iadd__(self, other):
        self.dim_check(other)
        if isinstance(other, FhePlainTensor):
            self.ep_add_unsafe(other)

        if isinstance(other, FheEncTensor):
            self.ee_add_unsafe(other)

        return self

    def __isub__(self, other):
        self.dim_check(other)
        if isinstance(other, FhePlainTensor):
            self.ep_sub_unsafe(other)

        if isinstance(other, FheEncTensor):
            self.ee_sub_unsafe(other)

        return self

    def __neg__(self):
        for i in range(self.num_batch):
            self.evaluator.negate_inplace(self.cts[i])
        return self


class FheBuilder(object):
    secret_key = None
    public_key = None
    galois_keys = None
    encryptor = None
    decryptor = None

    def __init__(self, modulus, degree):
        self.modulus = modulus
        self.degree = degree
        self.acceptable_degree = [1024, 2048, 4096, 8192, 16384]
        if self.degree not in self.acceptable_degree:
            raise Exception(f"Get {self.degree}, but expect degree {self.acceptable_degree}")

        self.parms = EncryptionParameters(scheme_type.BFV)
        self.parms.set_poly_modulus_degree(self.degree)
        self.parms.set_coeff_modulus(CoeffModulus.BFVDefault(self.degree))
        # self.parms.set_coeff_modulus(CoeffModulus.Create(self.degree, [60]))
        self.parms.set_plain_modulus(self.modulus)
        # print(self.parms.coeff_modulus()[0].value())
        self.context = SEALContext.Create(self.parms)
        self.keygen = KeyGenerator(self.context)
        self.evaluator = Evaluator(self.context)
        self.batch_encoder = BatchEncoder(self.context)

    def generate_keys(self):
        self.secret_key = self.keygen.secret_key()
        self.public_key = self.keygen.public_key()
        self.encryptor = Encryptor(self.context, self.public_key)
        self.decryptor = Decryptor(self.context, self.secret_key)

    def generate_galois_keys(self):
        self.galois_keys = self.keygen.galois_keys()

    def get_public_key_buffer(self) -> torch.Tensor:
        if self.public_key is None:
            self.public_key = self.keygen.public_key()
        return torch_from_buffer(self.public_key)

    def build_from_loaded_public_key(self):
        self.encryptor = Encryptor(self.context, self.public_key)

    def get_secret_key_buffer(self) -> torch.Tensor:
        if self.secret_key is None:
            self.secret_key = self.keygen.secret_key()
        return torch_from_buffer(self.secret_key)

    def build_from_loaded_secret_key(self):
        self.decryptor = Decryptor(self.context, self.secret_key)

    def build_enc(self, num_elem, is_cheap_init=False) -> FheEncTensor:
        return FheEncTensor(num_elem, self.modulus, self.degree,
                            self.context, self.evaluator, self.batch_encoder, self.encryptor,
                            is_cheap_init=is_cheap_init)

    def build_enc_from_torch(self, torch_tensor) -> FheEncTensor:
        res = self.build_enc(get_prod(torch_tensor.size()))
        res.encrypt_additive(torch_tensor)
        return res

    def build_plain(self, num_elem) -> FhePlainTensor:
        return FhePlainTensor(num_elem, self.modulus, self.degree, self.context, self.evaluator, self.batch_encoder)

    def build_plain_from_torch(self, torch_tensor) -> FhePlainTensor:
        res = self.build_plain(get_prod(torch_tensor.size()))
        res.load_from_torch(torch_tensor)
        return res

    def decrypt_to_torch(self, fhe_enc_tensor: FheEncTensor, dst=None, to="gpu") -> torch.Tensor:
        def sub_decrypt(enc):
            return enc.decrypt(self.decryptor, dst=dst, to=to)

        if isinstance(fhe_enc_tensor, FheEncTensor):
            return sub_decrypt(fhe_enc_tensor)
        elif isinstance(fhe_enc_tensor, list):
            return torch.stack(list(map(sub_decrypt, fhe_enc_tensor)))
        else:
            raise Exception(f"Unexpected type of fhe_enc_tensor: {fhe_enc_tensor}")

    def noise_budget(self, fhe_enc_tensor: FheEncTensor, name=None):
        noise_bits = self.decryptor.invariant_noise_budget(fhe_enc_tensor.cts[0])
        if name is not None:
            print(f"{name} noise budget {noise_bits} bits")
        return noise_bits


def sub_handle(func, enc, another_arg=None):
    if isinstance(enc, FheEncTensor):
        if another_arg is None:
            return func(enc)
        else:
            return func(enc, another_arg)
    elif isinstance(enc, list):
        if not isinstance(enc[0], FheEncTensor):
            raise Exception(f"Unknown element type: {type(enc[0])}")
        if another_arg is None:
            return list(map(func, enc))
        else:
            return [func(enc[i], another_arg[i]) for i in range(len(enc))]
    else:
        raise Exception(f"Unknown type: {type(enc)}")


def test_cipher():
    print()
    print("Test for Cipher: start")
    modulus, degree = 12289, 2048
    num_elem = 2 ** 17
    print(f"modulus: {modulus}, degree: {degree}")

    fhe_builder = FheBuilder(modulus, degree)
    fhe_builder.generate_keys()
    tensor = torch.ones(num_elem).float() + 5
    enc_old = fhe_builder.build_enc_from_torch(tensor)
    ct_new = Ciphertext(enc_old.cts[1])
    print(torch_from_buffer(enc_old.cts[1]))
    print(torch_from_buffer(ct_new))
    torch_from_buffer(enc_old.cts[1])[0] += 1
    print(torch_from_buffer(enc_old.cts[1]))
    print(torch_from_buffer(ct_new))

    print("Test for FheBuilder rebuild: end")
    print()


def test_fhe_enc_copy():
    print()
    print("Test for FheEncTensor Copy: start")
    modulus, degree = 12289, 2048
    num_elem = 2 ** 17
    print(f"modulus: {modulus}, degree: {degree}")

    fhe_builder = FheBuilder(modulus, degree)
    fhe_builder.generate_keys()
    tensor = torch.ones(num_elem).float() + 5
    enc_old = fhe_builder.build_enc_from_torch(tensor)
    enc_new = enc_old.copy()
    actual = fhe_builder.decrypt_to_torch(enc_new)
    compare_expected_actual(tensor, actual, get_relative=True, name="fhe enc copy")

    print("Test for FheEncTensor Copy: end")
    print()


def test_fhe_builder_rebuild():
    print()
    print("Test for FheBuilder rebuild: start")
    modulus, degree = 12289, 2048
    num_elem = 2 ** 17
    print(f"modulus: {modulus}, degree: {degree}")

    fhe_builder_client = FheBuilder(modulus, degree)
    fhe_builder_server = FheBuilder(modulus, degree)
    fhe_builder_client.generate_keys()

    tensor_server = torch.ones(num_elem).float() + 5
    fhe_builder_server.get_public_key_buffer().copy_(fhe_builder_client.get_public_key_buffer())
    fhe_builder_server.build_from_loaded_public_key()
    enc_pk = fhe_builder_server.build_enc_from_torch(tensor_server)
    tensor_client = fhe_builder_client.decrypt_to_torch(enc_pk)
    compare_expected_actual(tensor_server, tensor_client, get_relative=True, name="Rebuilding public key")

    tensor_client_sk = torch.ones(num_elem).float() + 12
    fhe_builder_server.get_secret_key_buffer().copy_(fhe_builder_client.get_secret_key_buffer())
    fhe_builder_server.build_from_loaded_secret_key()
    enc_sk = fhe_builder_client.build_enc_from_torch(tensor_client_sk)
    tensor_server_sk = fhe_builder_server.decrypt_to_torch(enc_sk)
    compare_expected_actual(tensor_client_sk, tensor_server_sk, get_relative=True, name="Rebuilding secret key")

    print("Test for FheBuilder rebuild: end")
    print()


def test_fhe_builder():
    print()
    print("Test for FheBuilder: start")
    modulus, degree = 12289, 2048
    # modulus, degree = 65537, 2048
    # modulus, degree = 786433, 4096
    # modulus, degree = 65537, 4096
    num_elem = 2 ** 17 - 1
    fhe_builder = FheBuilder(modulus, degree)
    fhe_builder.generate_keys()
    print(f"modulus: {modulus}, degree: {degree}")
    print()

    gpu_tensor = gen_unirand_int_grain(0, modulus - 1, num_elem)
    gpu_tensor_rev = gen_unirand_int_grain(0, modulus - 1, num_elem)
    with NamedTimerInstance(f"build_plain_from_torch with num_elem: {num_elem}"):
        plain = fhe_builder.build_plain_from_torch(gpu_tensor)
    with NamedTimerInstance(f"plain.export_as_torch_gpu() with num_elem: {num_elem}"):
        tensor_from_plain = plain.export_as_torch()
    print("Fhe Plain encrypt and decrypt: ", end="")
    assert(compare_expected_actual(gpu_tensor, tensor_from_plain, verbose=True, get_relative=True).RelAvgDiff == 0)
    print()

    with NamedTimerInstance(f"fhe_builder.build_enc with num_elem: {num_elem}"):
        cipher = fhe_builder.build_enc(num_elem)
    with NamedTimerInstance(f"cipher.encrypt_additive with num_elem: {num_elem}"):
        cipher.encrypt_additive(gpu_tensor)
    with NamedTimerInstance(f"fhe_builder.decrypt_to_torch with num_elem: {num_elem}"):
        tensor_from_cipher = fhe_builder.decrypt_to_torch(cipher)
    print("Fhe Enc encrypt and decrypt: ", end="")
    assert(compare_expected_actual(gpu_tensor, tensor_from_cipher, verbose=True, get_relative=True).RelAvgDiff == 0)
    print()

    pt = fhe_builder.build_plain_from_torch(gpu_tensor)
    ct = fhe_builder.build_enc_from_torch(gpu_tensor_rev)
    with NamedTimerInstance(f"EP add with num_elem: {num_elem}"):
        ct += pt
    expected = pmod(gpu_tensor + gpu_tensor_rev, modulus)
    actual = fhe_builder.decrypt_to_torch(ct)
    assert(compare_expected_actual(expected, actual, verbose=True, get_relative=True).RelAvgDiff == 0)
    print()

    ct1 = fhe_builder.build_enc_from_torch(gpu_tensor)
    ct2 = fhe_builder.build_enc_from_torch(gpu_tensor_rev)
    with NamedTimerInstance(f"EE add with num_elem: {num_elem}"):
        ct1 += ct2
    expected = pmod(gpu_tensor + gpu_tensor_rev, modulus)
    actual = fhe_builder.decrypt_to_torch(ct1)
    assert(compare_expected_actual(expected, actual, verbose=True, get_relative=True).RelAvgDiff == 0)
    print()

    pt = fhe_builder.build_plain_from_torch(gpu_tensor)
    ct = fhe_builder.build_enc_from_torch(gpu_tensor_rev)
    with NamedTimerInstance(f"EP sub with num_elem: {num_elem}"):
        ct -= pt
    expected = pmod(gpu_tensor_rev - gpu_tensor, modulus)
    actual = fhe_builder.decrypt_to_torch(ct)
    assert(compare_expected_actual(expected, actual, verbose=True, get_relative=True).RelAvgDiff == 0)
    print()

    ct1 = fhe_builder.build_enc_from_torch(gpu_tensor)
    ct2 = fhe_builder.build_enc_from_torch(gpu_tensor_rev)
    with NamedTimerInstance(f"EE add with num_elem: {num_elem}"):
        ct1 -= ct2
    expected = pmod(gpu_tensor - gpu_tensor_rev, modulus)
    actual = fhe_builder.decrypt_to_torch(ct1)
    assert(compare_expected_actual(expected, actual, verbose=True, get_relative=True).RelAvgDiff == 0)
    print()

    pt = fhe_builder.build_plain_from_torch(gpu_tensor)
    ct = fhe_builder.build_enc_from_torch(gpu_tensor_rev)
    fhe_builder.noise_budget(ct, name="before ep")
    with NamedTimerInstance(f"EP Mult with num_elem: {num_elem}"):
        ct *= pt
    fhe_builder.noise_budget(ct, name="after ep")
    expected = pmod(gpu_tensor.double() * gpu_tensor_rev.double(), modulus)
    actual = fhe_builder.decrypt_to_torch(ct)
    assert(compare_expected_actual(expected, actual, verbose=True, get_relative=True).RelAvgDiff == 0)
    print()

    ct1 = fhe_builder.build_enc_from_torch(gpu_tensor)
    ct2 = fhe_builder.build_enc_from_torch(gpu_tensor_rev)
    fhe_builder.noise_budget(ct1, name="before ep")
    with NamedTimerInstance(f"EE mult with num_elem: {num_elem}"):
        ct1 *= ct2
    fhe_builder.noise_budget(ct1, name="before ep")
    expected = pmod(gpu_tensor.double() * gpu_tensor_rev.double(), modulus).float()
    actual = fhe_builder.decrypt_to_torch(ct1)
    assert(compare_expected_actual(expected, actual, verbose=True, get_relative=True).RelAvgDiff == 0)
    print()

    ct1 = fhe_builder.build_enc_from_torch(gpu_tensor)
    ct2 = fhe_builder.build_enc_from_torch(gpu_tensor_rev)
    fhe_builder.noise_budget(ct1, name="before ep")
    with NamedTimerInstance(f"EE Add with num_elem: {num_elem}"):
        ct1 *= ct2
    fhe_builder.noise_budget(ct1, name="before ep")
    expected = pmod(gpu_tensor.double() * gpu_tensor_rev.double(), modulus)
    actual = fhe_builder.decrypt_to_torch(ct1)
    assert(compare_expected_actual(expected, actual, verbose=True, get_relative=True).RelAvgDiff == 0)
    print()

    ct = fhe_builder.build_enc_from_torch(gpu_tensor)
    fhe_builder.noise_budget(ct, name="before ep")
    with NamedTimerInstance(f"neg E with num_elem: {num_elem}"):
        ct = -ct
    fhe_builder.noise_budget(ct, name="before ep")
    expected = pmod(-gpu_tensor, modulus)
    actual = fhe_builder.decrypt_to_torch(ct)
    assert(compare_expected_actual(expected, actual, verbose=True, get_relative=True).RelAvgDiff == 0)
    print()

    print("Test for FheBuilder: Finish")


def primesfrom2to(n):
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n//3 + (n%6==2), dtype=np.bool)
    sieve[0] = False
    for i in range(int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[      ((k*k)//3)      ::2*k] = False
            sieve[(k*k+4*k-2*k*(i&1))//3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0]+1)|1)]


def test_search_modulus():
    degree = 2048
    upper_limit = 2 ** 22
    plist = primesfrom2to(upper_limit)
    cond = np.mod(plist, 2 * degree) == 1
    plist = np.extract(cond, plist)
    p = plist[-1]
    for p in plist[::-1]:
        fhe_builder = FheBuilder(p, degree)
        fhe_builder.generate_keys()
        ori = [1, 2, 3, 2**14-3]
        enc = fhe_builder.build_enc_from_torch(torch.tensor(ori))
        dec = fhe_builder.decrypt_to_torch(enc)
        print(f"prime: {p}, smaller than: {upper_limit}")
        # compare_expected_actual(ori, dec, get_relative=True, name="test_search_modulus", verbose=True)
        if compare_expected_actual(ori, dec, get_relative=True, name="test_search_modulus").AvgRelDiff == 0:
            break


def test_noise():
    print()
    print("Test for FheBuilder: start")
    modulus, degree = 12289, 2048
    num_elem = 2 ** 14
    fhe_builder = FheBuilder(modulus, degree)
    fhe_builder.generate_keys()
    print(f"modulus: {modulus}, degree: {degree}")
    print()

    gpu_tensor = gen_unirand_int_grain(0, 2, num_elem)
    print(gpu_tensor)
    gpu_tensor_rev = gen_unirand_int_grain(0, modulus - 1, num_elem)
    with NamedTimerInstance(f"build_plain_from_torch with num_elem: {num_elem}"):
        plain = fhe_builder.build_plain_from_torch(gpu_tensor)
    with NamedTimerInstance(f"plain.export_as_torch_gpu() with num_elem: {num_elem}"):
        tensor_from_plain = plain.export_as_torch()
    print("Fhe Plain encrypt and decrypt: ", end="")
    assert(compare_expected_actual(gpu_tensor, tensor_from_plain, verbose=True, get_relative=True).RelAvgDiff == 0)
    print()

    with NamedTimerInstance(f"fhe_builder.build_enc with num_elem: {num_elem}"):
        cipher = fhe_builder.build_enc(num_elem)
    with NamedTimerInstance(f"cipher.encrypt_additive with num_elem: {num_elem}"):
        cipher.encrypt_additive(gpu_tensor)
    with NamedTimerInstance(f"fhe_builder.decrypt_to_torch with num_elem: {num_elem}"):
        tensor_from_cipher = fhe_builder.decrypt_to_torch(cipher)
    print("Fhe Enc encrypt and decrypt: ", end="")
    assert(compare_expected_actual(gpu_tensor, tensor_from_cipher, verbose=True, get_relative=True).RelAvgDiff == 0)
    print()

    pt = fhe_builder.build_plain_from_torch(gpu_tensor)
    ct = fhe_builder.build_enc_from_torch(gpu_tensor_rev)
    fhe_builder.noise_budget(ct, name="before ep")
    with NamedTimerInstance(f"EP Mult with num_elem: {num_elem}"):
        ct *= pt
    fhe_builder.noise_budget(ct, name="after ep")
    expected = pmod(gpu_tensor.double() * gpu_tensor_rev.double(), modulus)
    actual = fhe_builder.decrypt_to_torch(ct)
    assert(compare_expected_actual(expected, actual, verbose=True, get_relative=True).RelAvgDiff == 0)
    print()

    print("Test for FheBuilder: Finish")

if __name__ == "__main__":
    # test_fhe_builder()
    # test_fhe_builder_rebuild()
    # test_cipher()
    # test_fhe_enc_copy()
    test_search_modulus()
    # test_noise()

