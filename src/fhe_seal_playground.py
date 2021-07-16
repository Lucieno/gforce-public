import sys
import time
import io
import random

import numpy as np
from seal import *
from seal_helper import *


def rand_int():
    return int(random.random()*(10**10))


def bfv_comm(context):
    print_parameters(context)

    parms = context.first_context_data().parms()
    plain_modulus = parms.plain_modulus()
    poly_modulus_degree = parms.poly_modulus_degree()

    print("Generating secret/public keys: ", end="")
    keygen = KeyGenerator(context)
    print("Done")

    secret_key = keygen.secret_key()
    public_key = keygen.public_key()

    print("going to get sk data")
    sk_pt = secret_key.data()
    np_sk_pt = np.array(sk_pt, copy=False)
    print("np_sk_pt", np_sk_pt)
    print(len(np_sk_pt))

    print("going to get pk data")
    pk_ct = public_key.data()
    np_pk_ct = np.array(pk_ct, copy=False)
    print("np_pk_ct", np_pk_ct)
    print(len(np_pk_ct))

    encryptor = Encryptor(context, public_key)
    decryptor = Decryptor(context, secret_key)
    evaluator = Evaluator(context)
    batch_encoder = BatchEncoder(context)
    encoder = IntegerEncoder(context)

    print(evaluator)
    another_eva = evaluator
    print(another_eva)

    # These will hold the total times used by each operation.
    time_batch_sum = 0
    time_unbatch_sum = 0
    time_encrypt_sum = 0
    time_decrypt_sum = 0

    # Populate a vector of values to batch.
    slot_count = batch_encoder.slot_count()
    pod_vector = uIntVector()
    pod_vector2 = uIntVector()
    for i in range(slot_count):
        pod_vector.push_back(rand_int() % plain_modulus.value())
    print("Running tests ")

    arr = np.arange(2048).astype(np.uint64) + 2
    arr_lst = [np.arange(2048).astype(np.uint64) + i for i in range(64)]
    arr_pod = np.array(pod_vector, copy=False)
    arr_decode = np.ones(2048).astype(np.uint64)
    pod_vector2.from_np(arr)
    print(pod_vector2[5])

    t0 = time.time()
    for i in range(64):
        arr_lst[i] = arr_pod[:]
        # pod_vector2.from_np(arr_lst[i])
    t1 = time.time()
    print(f"time for copy: {(t1 - t0) * 10 ** 6} ns")

    plain_direct = Plaintext(parms.poly_modulus_degree(), 0)
    decode_vector = uIntVector()



    '''
    [Batching]
    There is nothing unusual here. We batch our random plaintext matrix
    into the polynomial. Note how the plaintext we create is of the exactly
    right size so unnecessary reallocations are avoided.
    '''
    plain = Plaintext(parms.poly_modulus_degree(), 0)
    plain2 = Plaintext(parms.poly_modulus_degree(), 0)
    time_start = time.time()
    batch_encoder.encode(pod_vector, plain)
    batch_encoder.encode(pod_vector2, plain2)
    time_end = time.time()
    time_batch_sum += (time_end - time_start) * 1000000

    plain_recv = Plaintext(parms.poly_modulus_degree(), 0)
    t0 = time.time()
    plain_str = plain.save_str()
    plain_recv.load_str(context, plain_str)
    t1 = time.time()
    print()
    np_plain = np.array(plain, copy=False)
    np_plain2 = np.array(plain2, copy=False)
    print(np_plain)
    print(np_plain2)
    np_plain2[:] = np_plain
    print(f"time of transfering to str: {(t1 - t0) * 10 ** 6} ns")
    print("len(plain_str)", len(plain_str))
    print("len(np_plain)", len(np_plain))

    '''
    [Unbatching]
    We unbatch what we just batched.
    '''
    pod_vector2 = uIntVector()
    time_start = time.time()
    batch_encoder.decode(plain, pod_vector2)
    time_end = time.time()
    time_unbatch_sum += (time_end - time_start) * 1000000
    for j in range(slot_count):
        if pod_vector[j] != pod_vector2[j]:
            raise Exception("Batch/unbatch failed. Something is wrong.")

    '''
    [Encryption]
    We make sure our ciphertext is already allocated and large enough
    to hold the encryption with these encryption parameters. We encrypt
    our random batched matrix here.
    '''
    encrypted = Ciphertext()
    time_start = time.time()

    encryptor.encrypt(plain, encrypted)
    time_end = time.time()
    time_encrypt_sum += (time_end - time_start) * 1000000

    '''
    [Decryption]
    We decrypt what we just encrypted.
    '''
    plain2 = Plaintext(poly_modulus_degree, 0)
    time_start = time.time()
    decryptor.decrypt(encrypted, plain2)
    time_end = time.time()
    time_decrypt_sum += (time_end - time_start) * 1000000
    if plain.to_string() != plain2.to_string():
        raise Exception("Encrypt/decrypt failed. Something is wrong.")

    np_encrypted = np.array(encrypted, copy=False)
    print(np_encrypted)
    print(len(np_encrypted))
    encrypted_recv = Ciphertext()


def example_bfv_comm():
    parms = EncryptionParameters(scheme_type.BFV)
    poly_modulus_degree = 2048
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
    parms.set_plain_modulus(786433)
    bfv_comm(SEALContext.Create(parms))


if __name__ == '__main__':
    print_example_banner("Example: Performance Test")

    example_bfv_comm()
