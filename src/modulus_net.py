import os
import sys
import time
from collections import namedtuple
import math
from typing import Dict

import re
from sympy.physics.units import bits
from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from logger_utils import Logger
from model_reading import get_last_conv_name, get_last_fc_name, get_net_config_name
from swalp_net import add_r_, NamedParam
from torch_utils import nmod, get_prod, pmod, argparser_distributed, MetaTruncRandomGenerator

first_layer_name = "conv1"
last_conv_name = "conv13"
last_fc_name = "fc3"

modulus = 8273921
modulus = 7340033
# modulus = (2**22 - 17)

def data_shift(data, config):
    bits = 8
    input_exp, _ = config
    exp = -input_exp + (bits - 2)
    res = shift_by_exp_plain(data, exp)
    res.clamp_(-2 ** (bits - 1), 2 ** (bits - 1) - 1)

    return res

def get_next_layer_name(layer_name, store_configs):
    # if layer_name == last_conv_name:
    if layer_name == get_last_conv_name(store_configs):
        return "fc1"
    else:
        digits = int(''.join(c for c in layer_name if c.isdigit()))
        if layer_name[:2] == "fc":
            pre_name = "fc"
        elif layer_name[:4] == "conv":
            pre_name = "conv"
        else:
            raise Exception(f"Unknown kind of layer names: {layer_name}")
        next_layer_name = pre_name + str(digits + 1)
        if layer_name == last_conv_name:
            next_layer_name = "fc1"
        # return next_layer_name + "Forward"
        return next_layer_name

def shift_by_exp_plain(data, exp, mode="stochastic"):
    d = (2 ** -exp)
    x = data // d
    if mode == "stochastic":
        add_r_(x)
        x.floor_()
    elif mode == "nearest":
        x.round_()

    return x


def shift_by_exp(data, exp, mode="stochastic"):
    d = (2 ** -exp)

    p = modulus
    x = data

    # r = torch.zeros_like(x).uniform_(0, p-1).type(torch.int32).float()
    # r = torch.zeros_like(x).type(torch.int32).float()
    # n_elem = data.numel()
    # r = torch.arange(n_elem).cuda().reshape_as(x)

    # r = torch.from_numpy(np.random.uniform(0, p-1, size=x.numel())).cuda()\
    #     .type(torch.int32).type(torch.float).reshape(x.size())

    n_elem = data.numel()
    meta_rg = MetaTruncRandomGenerator()
    rg = meta_rg.get_rg("plain")
    r = rg.gen_uniform(n_elem, p).cuda().reshape_as(x)

    x = nmod(x, p)
    x = F.relu(x)
    # x = pmod(x, p)
    # return torch.floor(x/d)
    psum_xr = pmod(x+r, p)
    # print("(psum_xr < r):", torch.mean((psum_xr < r).float()).item())
    wrapped = nmod(psum_xr//d - r//d + p//d, p)
    unwrapped = nmod(psum_xr//d - r//d, p)
    # return unwrapped
    # x = unwrapped
    # x = F.relu(x)
    # return x
    x = torch.where(psum_xr < r, wrapped, unwrapped)

    return x

def post_layer_shift(data, layer_name, store_configs):
    bit, ebit = 8, 8

    layer_name = re.search('(conv|fc)\d+', layer_name).group()

    # print("get_last_fc_name(store_configs)", get_last_fc_name(store_configs))
    # if layer_name[:-7] == get_last_fc_name(store_configs):
    if layer_name == get_last_fc_name(store_configs):
    # if layer_name[:-7] == get_last_fc_name(store_configs):
            return data

    next_layer_name = get_next_layer_name(layer_name, store_configs)
    # print("store_configs.keys()", store_configs.keys())

    input_exp, _  = store_configs[layer_name + "ForwardX"]
    weight_exp, _ = store_configs[layer_name + "ForwardY"]
    next_input_exp, _ = store_configs[next_layer_name + "ForwardX"]

    exp = weight_exp + input_exp - next_input_exp - bit + 2

    res = shift_by_exp(data, exp)
    # print(exp, math.log2(torch.max(torch.abs(res))))

    return res


# Inherit from Function
class ModulusMatMulFunction(Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, store_configs, bias=None, layer_name=None):

        layer_op_name = layer_name + "Forward"

        output_q = input.mm(weight.t())
        output_q = output_q + bias
        output_q = nmod(output_q, modulus)
        output = post_layer_shift(output_q, layer_op_name, store_configs)

        return output

    @staticmethod
    def backward(ctx, grad_output, ):
        return None


class ModulusMatmul(nn.Module):
    dtype = torch.float
    device = torch.device("cuda:0")

    def __init__(self, input_features, output_features, bias=True, layer_name=None):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.layer_name = layer_name

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features).type(self.dtype).to(self.device))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features).type(self.dtype).to(self.device))
        else:
            self.register_parameter('bias', None)

    def set_store_configs(self, store_configs):
        self.store_configs = store_configs

    def forward(self, input):
        return ModulusMatMulFunction.apply(input, self.weight, self.store_configs, self.bias, self.layer_name)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, layer_name={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.layer_name)


# Inherit from Function
class ModulusConv2dFunction(Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, store_configs, bias=None, layer_name=None, pool=None):

        layer_op_name = layer_name + "Forward"

        output_q = F.conv2d(input, weight, bias, padding=1)
        #output_q = output_q + bias#unmodified
        output_q = nmod(output_q, modulus)
        if pool is not None:
            output_q = pool(output_q)
        output = post_layer_shift(output_q, layer_op_name, store_configs)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None


class ModulusConv2d(nn.Module):
    dtype = torch.float
    device = torch.device("cuda:0")
    def __init__(self, NumInputChannel, NumOutputChannel, FilterHW, padding=1, bias=True, layer_name=None, is_pool=False):
        super().__init__()
        self.NumInputChannel = NumInputChannel
        self.NumOutputChannel = NumOutputChannel
        self.FilterHW = FilterHW
        self.layer_name = layer_name

        if is_pool:
            self.pool = nn.MaxPool2d(2, 2)
        else:
            self.pool = None

        self.weight = nn.Parameter(
            torch.Tensor(self.NumOutputChannel, self.NumInputChannel, self.FilterHW, self.FilterHW)
                .type(self.dtype).to(self.device))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.NumOutputChannel).type(self.dtype).to(self.device))
        else:
            self.register_parameter('bias', None)

    def set_store_configs(self, store_configs):
        self.store_configs = store_configs

    def forward(self, input):
        return ModulusConv2dFunction.apply(input, self.weight, self.store_configs, self.bias, self.layer_name, self.pool)

    def extra_repr(self):
        return 'NumInputChannel={}, NumOutputChannel={}, FilterHW={}, bias={}, layer_name={}'.format(
            self.NumInputChannel, self.NumOutputChannel, self.FilterHW, self.bias is not None,
            self.layer_name)


class ModulusNet(nn.Module):
    def __init__(self, store_configs: dict, num_class=10):
        super().__init__()
        self.store_configs = store_configs

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = ModulusConv2d(3, 64, 3, layer_name="conv1", is_pool=True)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = ModulusConv2d(64, 128, 3, layer_name="conv2", is_pool=False)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = ModulusConv2d(128, 256, 3, layer_name="conv3")
        self.conv4 = ModulusConv2d(256, 256, 3, layer_name="conv4", is_pool=True)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = ModulusConv2d(256, 512, 3, layer_name="conv5")
        self.conv6 = ModulusConv2d(512, 512, 3, layer_name="conv6", is_pool=True)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv7 = ModulusConv2d(512, 512, 3, layer_name="conv7")
        self.conv8 = ModulusConv2d(512, 512, 3, layer_name="conv8", is_pool=True)
        # self.relu8 = nn.ReLU()
        # self.pool8 = nn.MaxPool2d(2, 2)
        # self.pool = nn.MaxPool2d(2, 2)
        self.NumElemFlatten = 512

        self.fc1 = ModulusMatmul(self.NumElemFlatten, 512, layer_name="fc1")
        self.fc2 = ModulusMatmul(512, 512, layer_name="fc2")
        self.fc3 = ModulusMatmul(512, num_class, layer_name="fc3")

        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4,
                            self.conv5, self.conv6, self.conv7, self.conv8]
        self.fc_layers = [self.fc1, self.fc2, self.fc3]
        self.linear_layers = self.conv_layers + self.fc_layers

        for layer in self.linear_layers:
            layer.set_store_configs(self.store_configs)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool2(self.relu2((self.conv2(x))))
        #
        # x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        # x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        # x = self.pool8(self.relu8(self.conv8(F.relu(self.conv7(x)))))

        x = self.conv1(x)
        x = (self.conv2(x))

        x = self.conv4(self.conv3(x))
        x = self.conv6(self.conv5(x))
        x = self.conv8(self.conv7(x))

        x = x.view(-1, self.NumElemFlatten)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def load_weight(self, state_dict):
        bits = 8
        for layer in self.linear_layers:
            weight_exp, _ = self.store_configs[layer.layer_name + "ForwardY"]
            exp = -weight_exp + (bits - 2)
            weight_f = state_dict[layer.layer_name + ".weight"]
            layer.weight.data = shift_by_exp_plain(weight_f, exp)

class ModulusNet_vgg16(nn.Module):
    def __init__(self, store_configs: dict, num_class=100):
        super().__init__()
        self.store_configs = store_configs

        self.input_layer = nn.Identity()
        #self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = ModulusConv2d(3, 64, 3, layer_name="conv1")
        self.conv2 = ModulusConv2d(64, 64, 3, layer_name="conv2", is_pool=True)

        self.conv3 = ModulusConv2d(64, 128, 3, layer_name="conv3")
        self.conv4 = ModulusConv2d(128, 128, 3, layer_name="conv4", is_pool=True)

        self.conv5 = ModulusConv2d(128, 256, 3, layer_name="conv5")
        self.conv6 = ModulusConv2d(256, 256, 3, layer_name="conv6")
        self.conv7 = ModulusConv2d(256, 256, 3, layer_name="conv7", is_pool=True)

        self.conv8 = ModulusConv2d(256, 512, 3, layer_name="conv8")
        self.conv9 = ModulusConv2d(512, 512, 3, layer_name="conv9")
        self.conv10 = ModulusConv2d(512, 512, 3, layer_name="conv10", is_pool=True)

        self.conv11 = ModulusConv2d(512, 512, 3, layer_name="conv11")
        self.conv12 = ModulusConv2d(512, 512, 3, layer_name="conv12")
        self.conv13 = ModulusConv2d(512, 512, 3, layer_name="conv13", is_pool=True)
        self.NumElemFlatten = 512

        self.fc1 = ModulusMatmul(self.NumElemFlatten, 512, layer_name="fc1")
        self.fc2 = ModulusMatmul(512, 512, layer_name="fc2")
        self.fc3 = ModulusMatmul(512, num_class, layer_name="fc3")

        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4,
                            self.conv5, self.conv6, self.conv7, self.conv8,
                            self.conv9, self.conv10, self.conv11, self.conv12,
                            self.conv13]
        self.fc_layers = [self.fc1, self.fc2, self.fc3]
        self.linear_layers = self.conv_layers + self.fc_layers

        for layer in self.linear_layers:
            layer.set_store_configs(self.store_configs)

    def forward_without_bn(self, x):
        x = self.input_layer(x)
        x = self.conv2(self.conv1(x))
        x = self.conv4(self.conv3(x))

        x = self.conv7(self.conv6(F.relu(self.conv5(x))))
        x = self.conv10(self.conv9(F.relu(self.conv8(x))))
        x = self.conv13(self.conv12(F.relu(self.conv11(x))))

        x = x.view(-1, self.NumElemFlatten)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def forward(self, x):
        return self.forward_without_bn(x)

    def load_weight_bias(self, state_dict):
        bits = 8
        for layer in self.linear_layers:
            input_exp, _ = self.store_configs[layer.layer_name + "ForwardX"]
            weight_exp, _ = self.store_configs[layer.layer_name + "ForwardY"]
            w_exp = -weight_exp + (bits - 2)
            exp = -weight_exp + (bits - 2) - input_exp + (bits - 2)
            weight_f = state_dict[layer.layer_name + ".weight"]
            bias_f = state_dict[layer.layer_name + ".bias"]
            layer.weight.data = shift_by_exp_plain(weight_f, w_exp)
            layer.bias.data = shift_by_exp_plain(bias_f, exp)

class ModulusNet_MiniONN(nn.Module):
    def __init__(self, store_configs: dict):
        super().__init__()
        self.store_configs = store_configs

        self.input_layer = nn.Identity()

        self.conv1 = ModulusConv2d(3, 64, 3, layer_name="conv1")
        self.conv2 = ModulusConv2d(64, 64, 3, layer_name="conv2", is_pool=True)

        self.conv3 = ModulusConv2d(64, 64, 3, layer_name="conv3")
        self.conv4 = ModulusConv2d(64, 64, 3, layer_name="conv4", is_pool=True)
        # self.pool2 = nn.AvgPool2d(2, 2)

        self.conv5 = ModulusConv2d(64, 64, 3, layer_name="conv5")
        self.conv6 = ModulusConv2d(64, 64, 3, layer_name="conv6")
        self.conv7 = ModulusConv2d(64, 16, 3, layer_name="conv7")
        self.NumElemFlatten = 1024
        self.fc1 = ModulusMatmul(self.NumElemFlatten, 10, layer_name="fc1")

        self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4,
                            self.conv5, self.conv6, self.conv7]
        self.fc_layers = [self.fc1]
        self.linear_layers = self.conv_layers + self.fc_layers

        for layer in self.linear_layers:
            layer.set_store_configs(self.store_configs)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.conv1(x)
        x = self.conv3(self.conv2(x))
        x = self.conv7(self.conv6(self.conv5(self.conv4(x))))

        x = x.view(-1, self.NumElemFlatten)
        x = self.fc1(x)
        return x

    def load_weight_bias(self, state_dict):
        bits = 8
        for layer in self.linear_layers:
            input_exp, _ = self.store_configs[layer.layer_name + "ForwardX"]
            weight_exp, _ = self.store_configs[layer.layer_name + "ForwardY"]
            w_exp = -weight_exp + (bits - 2)
            exp = -weight_exp + (bits - 2) - input_exp + (bits - 2)
            weight_f = state_dict[layer.layer_name + ".weight"]
            bias_f = state_dict[layer.layer_name + ".bias"]
            layer.weight.data = shift_by_exp_plain(weight_f, w_exp)
            layer.bias.data = shift_by_exp_plain(bias_f, exp)

def main(test_to_run, modnet):
    print("CUDA is available:", torch.cuda.is_available())
    # torch.backends.cudnn.deterministic = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    minibatch = 512
    device = torch.device("cuda:0")

    # model_name_base = test_to_run
    # signific_acc = "897"
    # loss_name_base = f"./model/{model_name_base}_{signific_acc}"

    net_state_name, config_name = get_net_config_name(test_to_run)

    net_state = torch.load(net_state_name)
    store_configs = np.load(config_name, allow_pickle="TRUE").item()

    net = modnet(store_configs)
    net.load_weight_bias(net_state)
    net.to(device)

    def modulus_net_input_transform(data):
        bits = 8
        input_exp, _ = store_configs[first_layer_name + "ForwardX"]
        exp = -input_exp + (bits - 2)
        res = shift_by_exp_plain(data, exp)
        res.clamp_(-2 ** (bits - 1), 2 ** (bits - 1) - 1)

        return res

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), normalize, modulus_net_input_transform]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch, shuffle=False, num_workers=2)

    sys.stdout = Logger()

    training_start = time.time()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %2.3f %%' % (100 * correct / total))

    elapsed_time = time.time() - training_start
    print("Elapsed time for Prediction", elapsed_time)


    print('Finished Testing for Accuracy')


if __name__ == "__main__":
    _, _, _, test_to_run = argparser_distributed()
    if test_to_run == "vgg":
        main(test_to_run, ModulusNet)
    if test_to_run == "minionn_maxpool":
        main(test_to_run, ModulusNet_MiniONN)
    #main()
