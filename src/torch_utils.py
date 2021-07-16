import argparse
from collections import namedtuple
from time import time

import numpy as np
import torch
from numpy.random.generator import default_rng

from torch.multiprocessing import Process

from config import Config

def load_state_dict(dir, model): 
    state_dict = torch.load(dir)
    device = torch.device("cuda:0")
    model.load_weight_bias(state_dict)
    model.to(device)

def load_swalp_state_dict(dir, model, withnorm): 
    checkpoint = torch.load(dir)
    state_dict = checkpoint['state_dict']
    model_names = list()
    own_state = model.state_dict()
    for name in own_state:
        if(withnorm or not "norm" in name):
            model_names.append(name)
    for name, swa_name in zip(model_names, state_dict):
        param = state_dict[swa_name]
        own_state[name].copy_(param)

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def gen_unirand_int_grain(inclusive_start, inclusive_end, size):
    return torch.from_numpy(np.random.uniform(inclusive_start, inclusive_end + 1, size=size)) \
        .type(torch.int32).type(torch.float)


def get_torch_size(shape):
    if isinstance(shape, (int, np.int32, np.int64)):
        return torch.Size([shape])
    elif isinstance(shape, list):
        return torch.Size(shape)
    elif isinstance(shape, torch.Size):
        return shape
    raise Exception(f"Unknown type of shape: {type(shape)}")


def get_num_byte(x: torch.Tensor):
    return x.nelement() * x.element_size()


def generate_random_mask(modulus, shape):
    shape = get_torch_size(shape)
    return gen_unirand_int_grain(0, modulus - 1, get_prod(shape)).reshape(shape)


def pmod(torch_tensor, modulus):
    return torch.remainder(torch_tensor, modulus)
    # return torch.fmod(torch.fmod(torch_tensor, modulus) + modulus, modulus)


def nmod(tensor: torch.Tensor, modulus: int) -> torch.Tensor:
    tensor = pmod(tensor, modulus)
    return torch.where(tensor < modulus//2, tensor, tensor - modulus)


def get_prod(x):
    return np.prod(list(x))


def get_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().detach().numpy()


def reject_outliers(data, m=2):
    return data[abs(data - torch.mean(data)) < m * torch.std(data)]


def compare_expected_actual(expected, actual, show_where_err=False,
                            get_relative=False, verbose=False, show_values=False, name=""):
    def purify(x):
        # return torch.tensor(x)
        res = x.reshape(-1)
        if not isinstance(x, torch.Tensor):
            res = torch.tensor(x)
            # return x.detach().numpy()
        return res.type(torch.float).to(Config.device)
    expected = purify(expected)
    actual = purify(actual)

    if show_values:
        print("expected:", expected[0, 0])
        print("actual:", actual[0, 0])

    avg_abs_diff = torch.mean(torch.abs(expected - actual)).item()
    res = avg_abs_diff

    if show_where_err:
        # show_indices = torch.abs(expected - actual) / expected > 0.5
        show_indices = (expected != actual)
        print("expected values:", expected[show_indices])
        print("actual values:", actual[show_indices])
        print("difference:", (expected - actual)[show_indices])

    if get_relative:
        tmp_expected, tmp_actual = expected[expected != 0], actual[expected != 0]
        relative_diff = torch.abs(tmp_expected - tmp_actual) / torch.abs(tmp_expected)
        relative_avg_diff = torch.mean(torch.abs(tmp_actual - tmp_expected)) / torch.mean(torch.abs(tmp_expected))
        Error = namedtuple("Error", ("AvgAbsDiff", "RelAvgDiff", "AvgRelDiff", "StdRelDiff"))
        res = Error(avg_abs_diff, relative_avg_diff.item(), torch.mean(relative_diff).item(), torch.std(relative_diff).item())

    if name != "":
        print(f"{name}:", res)
    elif verbose:
        print(res)

    return res


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def warming_up_cuda():
    # device = torch.device("cuda:0")
    device = torch.device(Config.device)

    print("Execution device: ", device)
    print("PyTorch version: ", torch.__version__)
    print("CUDA version: ", torch.version.cuda)
    if Config.use_cuda:
        print("CUDA device:", torch.cuda.get_device_name(0))

    seed = int(time()) if Config.global_random_seed is None else Config.global_random_seed
    print("The global seed set in warming up cuda is", seed)
    set_seed(seed)
    # set_seed(int(time()))
    # set_seed(123)

    a = torch.arange(10).double().to(Config.device)
    b = torch.arange(10).double().to(Config.device)
    c = a * b


def torch_from_buffer(buffer):
    # return torch.tensor(np.array(buffer, copy=False), dtype=torch.uint8)
    return torch.from_numpy(np.frombuffer(buffer, dtype=np.uint8))


def shuffle_torch(torch_tensor, shuffling_order):
    return torch.zeros_like(torch_tensor).scatter_(0, shuffling_order, torch_tensor.to(Config.device))


def torch_to_list(torch_tensor):
    assert(len(torch_tensor.size) == 2)
    return [torch_tensor[0] for i in torch_tensor.size[0]]


def marshal_funcs(funcs):
    procs = [Process(target=f) for f in funcs]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        p.terminate()


class RandomGenerator:
    rg: default_rng
    def __init__(self, seed):
        self.rg = default_rng(seed)

    def gen_uniform(self, n_elem: int, modulus: int):
        return torch.from_numpy(self.rg.uniform(0, modulus-1, size=n_elem)).type(torch.int32).type(torch.float)


# https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html
class MetaTruncRandomGeneratorBorg:
    _shared_state = {}
    def __init__(self):
        self.__dict__ = self._shared_state


class MetaTruncRandomGenerator(MetaTruncRandomGeneratorBorg):
    seed = None
    rgs = {}
    def __init__(self):
        MetaTruncRandomGeneratorBorg.__init__(self)
        if self.seed is None:
            self.reset_seed()

    def reset_seed(self):
        self.seed = int(time()) if Config.trunc_seed is None else Config.trunc_seed
        print("MetaTruncRandomGenerator set the seed to be:", self.seed)
        self.reset_rg("secure")
        self.reset_rg("plain")

    def reset_rg(self, name):
        print("MetaTruncRandomGenerator reset", name)
        self.rgs[name] = RandomGenerator(self.seed)

    def get_rg(self, name):
        return self.rgs[name]


def argparser_distributed():
    default_sid = Config.both_rank
    default_ip = "127.0.0.1"
    default_port = "29501"
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", "-s",
                        type=int,
                        default=default_sid,
                        help="The ID of the server")
    parser.add_argument("--ip",
                        dest="MasterAddr",
                        default=default_ip,
                        help="The Master Address for communication")
    parser.add_argument("--port",
                        dest="MasterPort",
                        default=default_port,
                        help="The Master Port for communication")
    parser.add_argument("--test", "-t",
                        dest="TestToRun",
                        default="all",
                        help="The Test to run")

    args = parser.parse_args()
    input_sid = args.sid
    MasterAddr = args.MasterAddr
    MasterPort = args.MasterPort
    test_to_run = args.TestToRun

    return input_sid, MasterAddr, MasterPort, test_to_run


