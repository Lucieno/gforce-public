from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from numpy import fft
from sympy import ntt, intt

from numbertheoretictransform import find_modulus, find_primitive_root
from timer_utils import NamedTimerInstance
from torch_utils import gen_unirand_int_grain, compare_expected_actual, pmod, get_prod


def calc_conv2d_output_shape(x_shape, w_shape, padding):
    batch_size = x_shape[0]
    n_output_channel = w_shape[0]
    img_hw = x_shape[3]
    filter_hw = w_shape[3]
    output_hw = img_hw + 2 * padding - (filter_hw - 1)
    return [batch_size, n_output_channel, output_hw, output_hw]


def transform2d(img, func, reverse=False):
    N = len(img)
    with NamedTimerInstance("Apply func ina loop"):
        tmp = [func(img[i, :]) for i in range(N)]
    with NamedTimerInstance("list to numpy"):
        img = np.array(tmp)
    img = np.array([func(img[:, i]) for i in range(N)])
    return img.transpose()


def get_pow_2_ceil(x):
    res = 1
    while res < x:
        res *= 2
    return res


def pad_to_size(x, padded_hw):
    img_hw = len(x)
    padded = np.zeros([padded_hw, padded_hw]).astype(np.int)
    padded[:img_hw, :img_hw] = x
    return padded


def pad_to_torch(x, padded_hw):
    img_hw = len(x)
    padded = torch.zeros([padded_hw, padded_hw]).double()
    padded[:img_hw, :img_hw] = x
    return padded


def pad_to_pow_2(x):
    return pad_to_size(x, get_pow_2_ceil(len(x)))


def test_basic_fft():
    def fft_from_1d(x2d):
        N = len(x2d)
        row_ffted = np.array([fft.fft(x2d[i, :]) for i in range(N)])
        fft2ded = np.array([fft.fft(row_ffted[:, i]) for i in range(N)])
        return fft2ded.transpose()

    x = np.array([[1, 9], [4, 3]])
    print(fft.fft2(x))
    print(fft_from_1d(x))


def test_basic_ntt():
    modulus = 786433
    img_hw = 34
    x = gen_unirand_int_grain(0, modulus - 1, img_hw ** 2).reshape([img_hw, img_hw]).numpy().astype(np.int)
    padded = np.zeros([get_pow_2_ceil(img_hw), get_pow_2_ceil(img_hw)]).astype(np.int)
    padded[:img_hw, :img_hw] = x
    x = padded
    expected = x[:, :]
    with NamedTimerInstance("NTT2D, Sympy"):
        ntted = transform2d(x, lambda sub_img: ntt(sub_img.tolist(), prime=modulus))
    with NamedTimerInstance("iNTT2D"):
        reved = transform2d(ntted, lambda sub_img: intt(sub_img.tolist(), prime=modulus))
    actual = reved
    compare_expected_actual(expected, actual, name="ntt", get_relative=True)


def test_nnt_conv_single_channel():
    modulus = 786433
    img_hw = 6
    filter_hw = 3
    padding = 1
    data_bit = 17
    data_range = 2 ** data_bit

    conv_hw = img_hw + 2 * padding
    padded_hw = get_pow_2_ceil(conv_hw)
    output_hw = img_hw + 2 * padding - (filter_hw - 1)
    output_offset = filter_hw - 2
    x = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, img_hw ** 2).reshape([img_hw, img_hw]).numpy().astype(np.int)
    w = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, filter_hw ** 2).reshape([filter_hw, filter_hw]).numpy().astype(np.int)
    x = np.arange(img_hw ** 2).reshape([img_hw, img_hw]).astype(np.int)
    w = np.arange(filter_hw ** 2).reshape([filter_hw, filter_hw]).astype(np.int)
    padded_x = pad_to_size(x, padded_hw)
    padded_w = pad_to_size(np.rot90(w, 2), padded_hw)
    print(padded_x)
    print(padded_w)
    with NamedTimerInstance("NTT2D, Sympy"):
        ntted_x = transform2d(padded_x, lambda sub_img: ntt(sub_img.tolist(), prime=modulus))
        ntted_w = transform2d(padded_w, lambda sub_img: ntt(sub_img.tolist(), prime=modulus))
    with NamedTimerInstance("Point-wise Dot"):
        doted = ntted_x * ntted_w
    with NamedTimerInstance("iNTT2D"):
        reved = transform2d(doted, lambda sub_img: intt(sub_img.tolist(), prime=modulus))
    actual = reved[output_offset:output_hw+output_offset, output_offset:output_hw+output_offset]
    print("reved\n", reved)
    print("actual\n", actual)

    torch_x = torch.tensor(x).reshape([1, 1, img_hw, img_hw])
    torch_w = torch.tensor(w).reshape([1, 1, filter_hw, filter_hw])
    expected = F.conv2d(torch_x, torch_w, padding=1)
    expected = pmod(expected.reshape(output_hw, output_hw), modulus)
    print("expected", expected)
    compare_expected_actual(expected, actual, name="ntt", get_relative=True)


def generate_ntt_param(modulus, len_vector, data_range):
    mod = find_modulus(len_vector, modulus)
    root = find_primitive_root(len_vector, mod - 1, mod)
    assert(data_range < mod)
    ntt_mat = torch.zeros([len_vector, len_vector]).double()
    inv_mat = torch.zeros([len_vector, len_vector]).double()
    inv_root = pow(len_vector, (mod - 1) - 1, mod)
    for i, j in product(range(len_vector), repeat=2):
        ntt_mat[i, j] = pow(root, i * j, mod)
        inv_mat[i, j] = (pow(root, (mod - 1) - i * j, mod) * inv_root) % mod
    # print(torch.mm(ntt_mat.double(), inv_mat.double()).fmod_(mod))
    return root, mod, ntt_mat, inv_mat


def ntt_mat2d(ntt_mat, mod, x):
    ntt_mat = ntt_mat.double()
    x = torch.mm(x, ntt_mat.t()).fmod_(mod)
    x = torch.mm(ntt_mat, x).fmod_(mod)
    return x


def test_torch_ntt():
    modulus = 786433
    img_hw = 64
    filter_hw = 3
    padding = 1
    data_bit = 17

    len_vector = img_hw
    data_range = 2 ** data_bit
    root, mod, ntt_mat, inv_mat = generate_ntt_param(modulus, len_vector, data_range)

    x = np.arange(img_hw ** 2).reshape([img_hw, img_hw]).astype(np.int)
    w = np.arange(filter_hw ** 2).reshape([filter_hw, filter_hw]).astype(np.int)
    x = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, img_hw ** 2).reshape([img_hw, img_hw]).double()
    w = gen_unirand_int_grain(-data_range // 2 + 1, data_range // 2, filter_hw ** 2).reshape([filter_hw, filter_hw]).double()

    ntt_mat = ntt_mat.double()
    inv_mat = inv_mat.double()

    with NamedTimerInstance("Mat NTT 2d"):
        ntted = ntt_mat2d(ntt_mat, mod, x)
    reved = ntt_mat2d(inv_mat, mod, ntted)
    expected = pmod(x, modulus).type(torch.int)
    actual  = pmod(reved, modulus).type(torch.int)
    compare_expected_actual(expected, actual, name="ntt", get_relative=True)



class NttMatmul(object):
    def __init__(self, modulus, len_vector, data_range):
        self.modulus = modulus
        self.len_vector = len_vector
        self.data_range = data_range
        root, mod, ntt_mat, inv_mat = generate_ntt_param(modulus, len_vector, data_range)
        self.root = root
        self.mod = mod
        self.ntt_mat = ntt_mat.double()
        self.inv_mat = inv_mat.double()

    def ntt2d(self, img):
        return ntt_mat2d(self.ntt_mat, self.mod, img)

    def intt2d(self, img):
        return ntt_mat2d(self.inv_mat, self.mod, img)


if __name__ == "__main__":
    # test_basic_fft()
    # test_basic_ntt()
    # test_nnt_conv_single_channel()
    test_torch_ntt()
