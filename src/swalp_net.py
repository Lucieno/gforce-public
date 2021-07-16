from __future__ import print_function

import os
import sys
import time
from collections import namedtuple
import math

from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from config import Config
from logger_utils import Logger
from torch_utils import argparser_distributed, nmod, adjust_learning_rate

dtype = torch.float
device = torch.device("cuda:0")

modulus = 8273921

minibatch = 256
n_classes = 10  # Create random Tensors to hold inputs and outputs
training_state = 0


def mod_move_down(x):
    return nmod(x, modulus)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


NamedParam = namedtuple("NamedParam", ("Name", "Param"))
store_configs = {}


def add_r_(data):
    r = torch.rand_like(data)
    data.add_(r)


def swalp_quantize(param, bits=8, mode="stochastic"):
    data = param.Param
    ebit = 8
    max_entry = torch.max(torch.abs(data)).item()
    if max_entry == 0: return data
    max_exponent = math.floor(math.log2(max_entry))
    max_exponent = min(max(max_exponent, -2 ** (ebit - 1)), 2 ** (ebit - 1) - 1)
    i = data * 2 ** (-max_exponent + (bits - 2))
    if mode == "stochastic":
        add_r_(i)
        i.floor_()
    elif mode == "nearest":
        i.round_()
    # if "Forward" in param.Name:
    #     print(f"Quantize {param.Name}: {(-max_exponent + (bits - 2))} ")
    #     if torch.any(i < (-2 ** (bits - 1))):
    #         print(f"{param.Name} has too small that close to -inf!!!!!!!!!!!!!!!")
    #     if torch.any(i > (2 ** (bits - 1) - 1)):
    #         print(f"{param.Name} has values too large!!!!!!!")
    i.clamp_(-2 ** (bits - 1), 2 ** (bits - 1) - 1)
    temp = i
    store_configs[param.Name] = (max_exponent, bits)
    return temp


def quantize_from_trained(param, bits=8, mode="stochastic"):
    data = param.Param
    ebit = 8
    max_exponent, bits = store_configs[param.Name]
    i = data * 2 ** (-max_exponent + (bits - 2))
    if mode == "stochastic":
        add_r_(i)
        i.floor_()
    elif mode == "nearest":
        i.round_()
    return i


def quantize_op(param, layer_op_name):
    if training_state == Config.training:
        return swalp_quantize(param, 8)
    else:
        return quantize_from_trained(param, 8)


def dequantize_op(param, layer_op_name):
    x_exp, x_bits = store_configs[layer_op_name + "X"]
    y_exp, y_bits = store_configs[layer_op_name + "Y"]
    # if "Forward" in layer_op_name:
    #     print(f"Dequantize {layer_op_name}: {(x_exp - (x_bits - 2) + y_exp - (y_bits - 2))} ")
    return param.Param * 2 ** (x_exp - (x_bits - 2) + y_exp - (y_bits - 2))


def pre_quantize(param_x, param_y, layer_op_name, iter_time):
    named_x = NamedParam(layer_op_name + "X", param_x)
    named_y = NamedParam(layer_op_name + "Y", param_y)

    x_q = quantize_op(named_x, layer_op_name)
    y_q = quantize_op(named_y, layer_op_name)

    return x_q, y_q


def post_quantize(param_z, layer_op_name, iter_time):
    # return param_z
    named_z = NamedParam(layer_op_name + "ZQ", param_z)
    z_f = dequantize_op(named_z, layer_op_name)
    return z_f


def string_to_tensor(s):
    return torch.tensor(np.frombuffer(s.encode("ascii", "ignore"), dtype=np.uint8))


def tensor_to_string(t):
    return "".join(map(chr, t))

swa_start = 200
lr_swa_ratio = 0.2
lr_init = 0.05
#https://github.com/stevenygd/SWALP/
# Learning rate schedule
def schedule(epoch):
    t = (epoch) / swa_start
    lr_ratio = 0.01
    if t <= 0.5:
       factor = 1.0
    elif t <= 0.9:
       factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    elif t < 1.0:
       factor = lr_ratio
    else:
       factor = lr_swa_ratio
    return lr_init * factor

# Inherit from Function
class QuantizeMatMulFunction(Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, layer_name=None, iter_time=0):

        layer_op_name = layer_name + "Forward"
        InputQ, WeightQ = pre_quantize(input, weight, layer_op_name, iter_time)

        OutputQ = InputQ.mm(WeightQ.t())
        OutputQ = mod_move_down(OutputQ)

        output = post_quantize(OutputQ, layer_op_name, iter_time)

        ctx.save_for_backward(input, weight, bias, string_to_tensor(layer_name), torch.tensor(iter_time))

        return output

    @staticmethod
    def backward(ctx, grad_output, ):
        input, weight, bias, layer_name, iter_time = ctx.saved_tensors
        layer_name = tensor_to_string(layer_name)
        #print(layer_name)
        iter_time = int(iter_time)
        input_grad = weight_grad = bias_grad = None

        if ctx.needs_input_grad[0]:
            layer_op_name = layer_name + "BackwardInput"
            OutputGradQ, WeightQ = pre_quantize(grad_output, weight, layer_op_name, iter_time)

            InputGradQ = OutputGradQ.mm(WeightQ)
            InputGradQ = mod_move_down(InputGradQ)

            input_grad = post_quantize(InputGradQ, layer_op_name, iter_time)

        if ctx.needs_input_grad[1]:
            layer_op_name = layer_name + "BackwardWeight"
            OutputGradQ, InputQ = pre_quantize(grad_output, input, layer_op_name, iter_time)

            WeightGradQ = OutputGradQ.t().mm(InputQ)
            WeightGradQ = mod_move_down(WeightGradQ)

            weight_grad = post_quantize(WeightGradQ, layer_op_name, iter_time)

        return input_grad, weight_grad, bias_grad, None, None


class QuantizeMatmul(nn.Module):
    def __init__(self, input_features, output_features, bias=None, layer_name=None):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.layer_name = layer_name

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features).type(dtype).to(device))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features).type(dtype).to(device))
        else:
            self.register_parameter('bias', None)

        self.iter_time = -1

        # Not a very smart way to initialize weights
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, input):
        self.iter_time += 1
        return QuantizeMatMulFunction.apply(input, self.weight, self.bias, self.layer_name, self.iter_time)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, layer_name={}, iter_time={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.layer_name, self.iter_time
        )


# Inherit from Function
class QuantizeConv2dFunction(Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, stride=1, padding=1, layer_name=None, iter_time=0):

        ctx.save_for_backward(input, weight, bias, string_to_tensor(layer_name), torch.tensor(iter_time))
        ctx.stride = stride
        ctx.padding = padding

        layer_op_name = layer_name + "Forward"
        InputQ, WeightQ = pre_quantize(input, weight, layer_op_name, iter_time)

        InputQ.requires_grad_(True)
        WeightQ.requires_grad_(True)

        torch.set_grad_enabled(True)
        OutputQ = F.conv2d(InputQ, WeightQ, stride=stride, padding=padding)
        # print("!!!!!OutputQ.grad_fn", InputQ.requires_grad, WeightQ.requires_grad, OutputQ.requires_grad)
        ctx.grad_fn = OutputQ.grad_fn
        torch.set_grad_enabled(False)
        OutputQ = mod_move_down(OutputQ)

        output = post_quantize(OutputQ, layer_op_name, iter_time)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, layer_name, iter_time = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        grad_fn = ctx.grad_fn
        layer_name = tensor_to_string(layer_name)
        iter_time = int(iter_time)
        input_grad = weight_grad = bias_grad = None

        if ctx.needs_input_grad[0]:
            layer_op_name = layer_name + "BackwardInput"
            OutputGradQ, WeightQ = pre_quantize(grad_output, weight, layer_op_name, iter_time)

        if ctx.needs_input_grad[1]:
            layer_op_name = layer_name + "BackwardWeight"
            OutputGradQ, InputQ = pre_quantize(grad_output, input, layer_op_name, iter_time)

        InputGradQ, WeightGradQ, _ = grad_fn(OutputGradQ)

        if ctx.needs_input_grad[0]:
            layer_op_name = layer_name + "BackwardInput"
            InputGradQ = mod_move_down(InputGradQ)
            input_grad = post_quantize(InputGradQ, layer_op_name, iter_time)

        if ctx.needs_input_grad[1]:
            layer_op_name = layer_name + "BackwardWeight"
            WeightGradQ = mod_move_down(WeightGradQ)
            weight_grad = post_quantize(WeightGradQ, layer_op_name, iter_time)

        #return input_grad, weight_grad, bias_grad, None, None
        return input_grad, weight_grad, bias_grad, None, None, None, None


class QuantizeConv2d(nn.Module):
    def __init__(self, NumInputChannel, NumOutputChannel, FilterHW, padding=1, stride=1, bias=None, layer_name=None):
        super().__init__()
        self.NumInputChannel = NumInputChannel
        self.NumOutputChannel = NumOutputChannel
        self.FilterHW = FilterHW
        self.layer_name = layer_name

        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.Tensor(self.NumOutputChannel, self.NumInputChannel, self.FilterHW, self.FilterHW)
                .type(dtype).to(device))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.NumOutputChannel).type(dtype).to(device))
        else:
            self.register_parameter('bias', None)

        self.iter_time = -1

        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        self.iter_time += 1
        return QuantizeConv2dFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.layer_name, self.iter_time)

    def extra_repr(self):
        return 'NumInputChannel={}, NumOutputChannel={}, FilterHW={}, bias={}, layer_name={}, iter_time={}'.format(
            self.NumInputChannel, self.NumOutputChannel, self.FilterHW, self.bias is not None,
            self.layer_name, self.iter_time
        )

class CTX(object):
    saved_tensors = None

    def __init__(self):
        pass

    def save_for_backward(self, *args):
        self.saved_tensors = args


# ctx = CTX()


class NetQ_vgg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = QuantizeConv2d(3, 64, 3, layer_name="conv1")
        self.norm1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = QuantizeConv2d(64, 128, 3, layer_name="conv2")
        self.norm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.NumElemFlatten = 8192

        self.conv3 = QuantizeConv2d(128, 256, 3, layer_name="conv3")
        self.norm3 = nn.BatchNorm2d(256)
        self.conv4 = QuantizeConv2d(256, 256, 3, layer_name="conv4")
        self.norm4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = QuantizeConv2d(256, 512, 3, layer_name="conv5")
        self.norm5 = nn.BatchNorm2d(512)
        self.conv6 = QuantizeConv2d(512, 512, 3, layer_name="conv6")
        self.norm6 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv7 = QuantizeConv2d(512, 512, 3, layer_name="conv7")
        self.norm7 = nn.BatchNorm2d(512)
        self.conv8 = QuantizeConv2d(512, 512, 3, layer_name="conv8")
        self.norm8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU()
        self.pool8 = nn.MaxPool2d(2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.NumElemFlatten = 512

        # self.normfc1 = nn.BatchNorm1d(512)
        # self.normfc2 = nn.BatchNorm1d(512)
        # self.normfc3 = nn.BatchNorm1d(n_classes)
        self.fc1 = QuantizeMatmul(self.NumElemFlatten, 512, layer_name="fc1")
        self.fc2 = QuantizeMatmul(512, 512, layer_name="fc2")
        self.fc3 = QuantizeMatmul(512, num_classes, layer_name="fc3")

    def forward_without_bn(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(self.relu2((self.conv2(x))))

        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.pool8(self.relu8(self.conv8(F.relu(self.conv7(x)))))

        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_with_bn(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.norm2(self.conv2(x))))

        x = self.pool(F.relu(self.norm4(self.conv4(F.relu(self.norm3(self.conv3(x)))))))
        x = self.pool(F.relu(self.norm6(self.conv6(F.relu(self.norm5(self.conv5(x)))))))
        x = self.pool8(self.relu8(self.norm8(self.conv8(F.relu(self.norm7(self.conv7(x)))))))

        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        return self.forward_without_bn(x)

class NetQ_vgg16(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = QuantizeConv2d(3, 64, 3, layer_name="conv1")
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = QuantizeConv2d(64, 64, 3, layer_name="conv2")
        self.norm2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = QuantizeConv2d(64, 128, 3, layer_name="conv3")
        self.norm3 = nn.BatchNorm2d(128)
        self.conv4 = QuantizeConv2d(128, 128, 3, layer_name="conv4")
        self.norm4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        #self.NumElemFlatten = 8192

        self.conv5 = QuantizeConv2d(128, 256, 3, layer_name="conv5")
        self.norm5 = nn.BatchNorm2d(256)
        self.conv6 = QuantizeConv2d(256, 256, 3, layer_name="conv6")
        self.norm6 = nn.BatchNorm2d(256)
        self.conv7 = QuantizeConv2d(256, 256, 3, layer_name="conv7")
        self.norm7 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv8 = QuantizeConv2d(256, 512, 3, layer_name="conv8")
        self.norm8 = nn.BatchNorm2d(512)
        self.conv9 = QuantizeConv2d(512, 512, 3, layer_name="conv9")
        self.norm9 = nn.BatchNorm2d(512)
        self.conv10 = QuantizeConv2d(512, 512, 3, layer_name="conv10")
        self.norm10 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv11 = QuantizeConv2d(512, 512, 3, layer_name="conv11")
        self.norm11 = nn.BatchNorm2d(512)
        self.conv12 = QuantizeConv2d(512, 512, 3, layer_name="conv12")
        self.norm12 = nn.BatchNorm2d(512)
        self.conv13 = QuantizeConv2d(512, 512, 3, layer_name="conv13")
        self.norm13 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.NumElemFlatten = 512

        # self.normfc1 = nn.BatchNorm1d(512)
        # self.normfc2 = nn.BatchNorm1d(512)
        # self.normfc3 = nn.BatchNorm1d(n_classes)
        self.fc1 = QuantizeMatmul(self.NumElemFlatten, 4096, layer_name="fc1")
        self.dp1 = nn.Dropout()
        self.fc2 = QuantizeMatmul(4096, 4096, layer_name="fc2")
        self.dp2 = nn.Dropout()
        self.fc3 = QuantizeMatmul(4096, num_classes, layer_name="fc3")

    def forward_without_bn(self, x):
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))

        x = self.pool3(F.relu(self.conv7(F.relu(self.conv6(F.relu(self.conv5(x)))))))
        x = self.pool4(F.relu(self.conv10(F.relu(self.conv9(F.relu(self.conv8(x)))))))
        x = self.pool5(F.relu(self.conv13(F.relu(self.conv12(F.relu(self.conv11(x)))))))

        x = x.view(-1, self.NumElemFlatten)
        x = self.dp1(F.relu(self.fc1(x)))
        x = self.dp2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def forward(self, x):
        return self.forward_without_bn(x)


class NetQ_res34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        #conv1
        self.conv1 = QuantizeConv2d(3, 64, 7, stride=2, padding=3, layer_name="conv1")
        self.norm = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        #conv2_x
        self.conv2 = QuantizeConv2d(64, 64, 3, layer_name="conv2")
        self.norm2 = nn. BatchNorm2d(64)
        self.conv3 = QuantizeConv2d(64, 64, 3, layer_name="conv3")
        self.norm3 = nn. BatchNorm2d(64)
        
        self.conv4 = QuantizeConv2d(64, 64, 3, layer_name="conv4")
        self.norm4 = nn. BatchNorm2d(64)
        self.conv5 = QuantizeConv2d(64, 64, 3, layer_name="conv5")
        self.norm5 = nn. BatchNorm2d(64)

        self.conv6 = QuantizeConv2d(64, 64, 3, layer_name="conv6")
        self.norm6 = nn. BatchNorm2d(64)
        self.conv7 = QuantizeConv2d(64, 64, 3, layer_name="conv7")
        self.norm7 = nn. BatchNorm2d(64)

        #conv3_x
        #downsample
        self.conv08 = QuantizeConv2d(64, 128, 1, stride=2, padding=0, layer_name="conv08")
        self.norm08 = nn. BatchNorm2d(128)

        self.conv8 = QuantizeConv2d(64, 128, 3, stride=2, layer_name="conv8")
        self.norm8 = nn. BatchNorm2d(128)
        self.conv9 = QuantizeConv2d(128, 128, 3, layer_name="conv9")
        self.norm9 = nn. BatchNorm2d(128)

        self.conv10 = QuantizeConv2d(128, 128, 3, layer_name="conv10")
        self.norm10 = nn. BatchNorm2d(128)
        self.conv11 = QuantizeConv2d(128, 128, 3, layer_name="conv11")
        self.norm11 = nn. BatchNorm2d(128)

        self.conv12 = QuantizeConv2d(128, 128, 3, layer_name="conv12")
        self.norm12 = nn. BatchNorm2d(128)
        self.conv13 = QuantizeConv2d(128, 128, 3, layer_name="conv13")
        self.norm13 = nn. BatchNorm2d(128)

        self.conv14 = QuantizeConv2d(128, 128, 3, layer_name="conv14")
        self.norm14 = nn. BatchNorm2d(128)
        self.conv15 = QuantizeConv2d(128, 128, 3, layer_name="conv15")
        self.norm15 = nn. BatchNorm2d(128)

        #conv4_x
        #downsample
        self.conv016 = QuantizeConv2d(128, 256, 1, stride=2, padding=0, layer_name="conv016")
        self.norm016 = nn. BatchNorm2d(256)

        self.conv16 = QuantizeConv2d(128, 256, 3, stride=2, layer_name="conv16")
        self.norm16 = nn. BatchNorm2d(256)
        self.conv17 = QuantizeConv2d(256, 256, 3, layer_name="conv17")
        self.norm17 = nn. BatchNorm2d(256)

        self.conv18 = QuantizeConv2d(256, 256, 3, layer_name="conv18")
        self.norm18 = nn.BatchNorm2d(256)
        self.conv19 = QuantizeConv2d(256, 256, 3, layer_name="conv19")
        self.norm19 = nn.BatchNorm2d(256)

        self.conv20 = QuantizeConv2d(256, 256, 3, layer_name="conv20")
        self.norm20 = nn.BatchNorm2d(256)
        self.conv21 = QuantizeConv2d(256, 256, 3, layer_name="conv21")
        self.norm21 = nn.BatchNorm2d(256)

        self.conv22 = QuantizeConv2d(256, 256, 3, layer_name="conv22")
        self.norm22 = nn.BatchNorm2d(256)
        self.conv23 = QuantizeConv2d(256, 256, 3, layer_name="conv23")
        self.norm23 = nn.BatchNorm2d(256)

        self.conv24 = QuantizeConv2d(256, 256, 3, layer_name="conv24")
        self.norm24 = nn.BatchNorm2d(256)
        self.conv25 = QuantizeConv2d(256, 256, 3, layer_name="conv25")
        self.norm25 = nn.BatchNorm2d(256)

        self.conv26 = QuantizeConv2d(256, 256, 3, layer_name="conv26")
        self.norm26 = nn.BatchNorm2d(256)
        self.conv27 = QuantizeConv2d(256, 256, 3, layer_name="conv27")
        self.norm27 = nn.BatchNorm2d(256)

        #conv5_x
        self.conv028 = QuantizeConv2d(256, 512, 1, stride=2, padding=0, layer_name="conv028")
        self.norm028 = nn.BatchNorm2d(512)
        self.conv28 = QuantizeConv2d(256, 512, 3, stride=2, layer_name="conv28")
        self.norm28 = nn.BatchNorm2d(512)
        self.conv29 = QuantizeConv2d(512, 512, 3, layer_name="conv29")
        self.norm29 = nn.BatchNorm2d(512)

        self.conv30 = QuantizeConv2d(512, 512, 3, layer_name="conv30")
        self.norm30 = nn.BatchNorm2d(512)
        self.conv31 = QuantizeConv2d(512, 512, 3, layer_name="conv31")
        self.norm31 = nn.BatchNorm2d(512)

        self.conv32 = QuantizeConv2d(512, 512, 3, layer_name="conv32")
        self.norm32 = nn.BatchNorm2d(512)
        self.conv33 = QuantizeConv2d(512, 512, 3, layer_name="conv33")
        self.norm33 = nn.BatchNorm2d(512)

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.NumElemFlatten = 512

        self.fc = QuantizeMatmul(512, num_classes, layer_name="fc")

    def forward(self, x):

        x = self.pool(F.relu(self.norm(self.conv1(x))))

        x = F.relu(self.norm3(self.conv3(F.relu(self.norm2(self.conv2(x))))) + x)
        x = F.relu(self.norm5(self.conv5(F.relu(self.norm4(self.conv4(x))))) + x)
        x = F.relu(self.norm7(self.conv7(F.relu(self.norm6(self.conv6(x))))) + x)

        identity = x
        x = self.norm9(self.conv9(F.relu(self.norm8(self.conv8(x)))))
        identity = self.norm08(self.conv08(identity))
        x = F.relu(x+identity)
        x = F.relu(self.norm11(self.conv11(F.relu(self.norm10(self.conv10(x))))) + x)
        x = F.relu(self.norm13(self.conv13(F.relu(self.norm12(self.conv12(x))))) + x)
        x = F.relu(self.norm15(self.conv15(F.relu(self.norm14(self.conv14(x))))) + x)

        identity = x
        x = self.norm17(self.conv17(F.relu(self.norm16(self.conv16(x)))))
        identity = self.norm016(self.conv016(identity))
        x = F.relu(x + identity)
        x = F.relu(self.norm19(self.conv19(F.relu(self.norm18(self.conv18(x))))) + x)
        x = F.relu(self.norm21(self.conv21(F.relu(self.norm20(self.conv20(x))))) + x)
        x = F.relu(self.norm23(self.conv23(F.relu(self.norm22(self.conv22(x))))) + x)
        x = F.relu(self.norm25(self.conv25(F.relu(self.norm24(self.conv24(x))))) + x)
        x = F.relu(self.norm27(self.conv27(F.relu(self.norm26(self.conv26(x))))) + x)

        identity = x   
        x = self.norm29(self.conv29(F.relu(self.norm28(self.conv28(x)))))
        identity = self.norm028(self.conv028(identity))
        x = F.relu(x + identity)
        x = F.relu(self.norm31(self.conv31(F.relu(self.norm30(self.conv30(x))))) + x)
        x = F.relu(self.norm33(self.conv33(F.relu(self.norm32(self.conv32(x))))) + x)
        x = self.pool2(x)

        x = x.view(-1, self.NumElemFlatten)
        x = self.fc(x)
        return x

class NetQ_vgg19(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = QuantizeConv2d(3, 64, 3, layer_name="conv1")
        self.norm1 = nn.BatchNorm2d(64)
        self.conv1 = QuantizeConv2d(3, 64, 3, layer_name="conv1")
        self.norm1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = QuantizeConv2d(64, 128, 3, layer_name="conv2")
        self.norm2 = nn.BatchNorm2d(128)
        self.conv2 = QuantizeConv2d(128, 128, 3, layer_name="conv2")
        self.norm2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        #self.NumElemFlatten = 8192

        self.conv3 = QuantizeConv2d(128, 256, 3, layer_name="conv3")
        self.norm3 = nn.BatchNorm2d(256)
        self.conv4 = QuantizeConv2d(256, 256, 3, layer_name="conv4")
        self.norm4 = nn.BatchNorm2d(256)
        self.conv4 = QuantizeConv2d(256, 256, 3, layer_name="conv4")
        self.norm4 = nn.BatchNorm2d(256)
        self.conv4 = QuantizeConv2d(256, 256, 3, layer_name="conv4")
        self.norm4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv5 = QuantizeConv2d(256, 512, 3, layer_name="conv5")
        self.norm5 = nn.BatchNorm2d(512)
        self.conv6 = QuantizeConv2d(512, 512, 3, layer_name="conv6")
        self.norm6 = nn.BatchNorm2d(512)
        self.conv6 = QuantizeConv2d(512, 512, 3, layer_name="conv6")
        self.norm6 = nn.BatchNorm2d(512)
        self.conv6 = QuantizeConv2d(512, 512, 3, layer_name="conv6")
        self.norm6 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv7 = QuantizeConv2d(512, 512, 3, layer_name="conv7")
        self.norm7 = nn.BatchNorm2d(512)
        self.conv8 = QuantizeConv2d(512, 512, 3, layer_name="conv8")
        self.norm8 = nn.BatchNorm2d(512)
        self.conv7 = QuantizeConv2d(512, 512, 3, layer_name="conv7")
        self.norm7 = nn.BatchNorm2d(512)
        self.conv8 = QuantizeConv2d(512, 512, 3, layer_name="conv8")
        self.norm8 = nn.BatchNorm2d(512)
        self.pool8 = nn.MaxPool2d(2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.NumElemFlatten = 512

        # self.normfc1 = nn.BatchNorm1d(512)
        # self.normfc2 = nn.BatchNorm1d(512)
        # self.normfc3 = nn.BatchNorm1d(n_classes)
        self.fc1 = QuantizeMatmul(self.NumElemFlatten, 4096, layer_name="fc1")
        self.fc2 = QuantizeMatmul(4096, 4096, layer_name="fc2")
        self.fc3 = QuantizeMatmul(4096, num_classes, layer_name="fc3")

    def forward_without_bn(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(self.relu2((self.conv2(x))))

        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.pool8(self.relu8(self.conv8(F.relu(self.conv7(x)))))

        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_with_bn(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.norm2(self.conv2(x))))

        x = self.pool(F.relu(self.norm4(self.conv4(F.relu(self.norm3(self.conv3(x)))))))
        x = self.pool(F.relu(self.norm6(self.conv6(F.relu(self.norm5(self.conv5(x)))))))
        x = self.pool8(self.relu8(self.norm8(self.conv8(F.relu(self.norm7(self.conv7(x)))))))

        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        return self.forward_without_bn(x)

class NetQ_imagenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = QuantizeConv2d(3, 64, 3, layer_name="conv1")
        self.norm1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = QuantizeConv2d(64, 128, 3, layer_name="conv2")
        self.norm2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        #self.NumElemFlatten = 8192

        self.conv3 = QuantizeConv2d(128, 256, 3, layer_name="conv3")
        self.norm3 = nn.BatchNorm2d(256)
        self.conv4 = QuantizeConv2d(256, 256, 3, layer_name="conv4")
        self.norm4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv5 = QuantizeConv2d(256, 512, 3, layer_name="conv5")
        self.norm5 = nn.BatchNorm2d(512)
        self.conv6 = QuantizeConv2d(512, 512, 3, layer_name="conv6")
        self.norm6 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv7 = QuantizeConv2d(512, 512, 3, layer_name="conv7")
        self.norm7 = nn.BatchNorm2d(512)
        self.conv8 = QuantizeConv2d(512, 512, 3, layer_name="conv8")
        self.norm8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU()
        self.pool8 = nn.MaxPool2d(2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.NumElemFlatten = 512

        # self.normfc1 = nn.BatchNorm1d(512)
        # self.normfc2 = nn.BatchNorm1d(512)
        # self.normfc3 = nn.BatchNorm1d(n_classes)
        self.fc1 = QuantizeMatmul(self.NumElemFlatten, 4096, layer_name="fc1")
        self.fc2 = QuantizeMatmul(4096, 4096, layer_name="fc2")
        self.fc3 = QuantizeMatmul(4096, 1000, layer_name="fc3")

    def forward_without_bn(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(self.relu2((self.conv2(x))))

        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.pool8(self.relu8(self.conv8(F.relu(self.conv7(x)))))

        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_with_bn(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(self.relu2(self.norm2(self.conv2(x))))

        x = self.pool(F.relu(self.norm4(self.conv4(F.relu(self.norm3(self.conv3(x)))))))
        x = self.pool(F.relu(self.norm6(self.conv6(F.relu(self.norm5(self.conv5(x)))))))
        x = self.pool8(self.relu8(self.norm8(self.conv8(F.relu(self.norm7(self.conv7(x)))))))

        x = x.view(-1, self.NumElemFlatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        return self.forward_without_bn(x)


class NetQ_MiniONN_maxpool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = QuantizeConv2d(3, 64, 3, layer_name="conv1")
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = QuantizeConv2d(64, 64, 3, layer_name="conv2")
        self.norm2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        #self.NumElemFlatten = 8192

        self.conv3 = QuantizeConv2d(64, 64, 3, layer_name="conv3")
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = QuantizeConv2d(64, 64, 3, layer_name="conv4")
        self.norm4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = QuantizeConv2d(64, 64, 3, layer_name="conv5")
        self.norm5 = nn.BatchNorm2d(64)
        self.conv6 = QuantizeConv2d(64, 64, 3, layer_name="conv6")
        self.norm6 = nn.BatchNorm2d(64)
        self.conv7 = QuantizeConv2d(64, 16, 3, layer_name="conv7")
        self.norm7 = nn.BatchNorm2d(16)

        self.NumElemFlatten = 1024

        # self.normfc1 = nn.BatchNorm1d(512)
        self.fc1 = QuantizeMatmul(self.NumElemFlatten, 10, layer_name="fc1")#???

    def forward_without_bn(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv3(self.pool1(F.relu((self.conv2(x))))))
        x = F.relu(self.conv7(F.relu(self.conv6(F.relu(self.conv5(self.pool2(F.relu(self.conv4(x)))))))))

        x = x.view(-1, self.NumElemFlatten)
        x = self.fc1(x)
        return x

    def forward_with_bn(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm3(self.conv3(self.pool1(F.relu(self.norm2(self.conv2(x)))))))
        x = F.relu(self.norm7(self.conv7(F.relu(self.norm6(self.conv6(F.relu(self.norm5(self.conv5(self.pool2(F.relu(self.norm4(self.conv4(x)))))))))))))

        x = x.view(-1, self.NumElemFlatten)
        x = self.fc1(x)
        return x

    def forward(self, x):
        return self.forward_without_bn(x)

class NetQ_MiniONN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = QuantizeConv2d(3, 64, 3, layer_name="conv1")
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = QuantizeConv2d(64, 64, 3, layer_name="conv2")
        self.norm2 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(2, 2)
        #self.NumElemFlatten = 8192

        self.conv3 = QuantizeConv2d(64, 64, 3, layer_name="conv3")
        self.norm3 = nn.BatchNorm2d(64)
        self.conv4 = QuantizeConv2d(64, 64, 3, layer_name="conv4")
        self.norm4 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.conv5 = QuantizeConv2d(64, 64, 3, layer_name="conv5")
        self.norm5 = nn.BatchNorm2d(64)
        self.conv6 = QuantizeConv2d(64, 64, 3, layer_name="conv6")
        self.norm6 = nn.BatchNorm2d(64)
        self.conv7 = QuantizeConv2d(64, 16, 3, layer_name="conv7")
        self.norm7 = nn.BatchNorm2d(16)

        self.NumElemFlatten = 1024

        # self.normfc1 = nn.BatchNorm1d(512)
        self.fc1 = QuantizeMatmul(self.NumElemFlatten, 10, layer_name="fc1")

    def forward_without_bn(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv3(self.pool1(F.relu((self.conv2(x))))))
        x = F.relu(self.conv7(F.relu(self.conv6(F.relu(self.conv5(self.pool2(F.relu(self.conv4(x)))))))))

        x = x.view(-1, self.NumElemFlatten)
        x = self.fc1(x)
        return x

    def forward_with_bn(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm3(self.conv3(self.pool1(F.relu(self.norm2(self.conv2(x)))))))
        x = F.relu(self.norm7(self.conv7(F.relu(self.norm6(self.conv6(F.relu(self.norm5(self.conv5(self.pool2(F.relu(self.norm4(self.conv4(x)))))))))))))

        x = x.view(-1, self.NumElemFlatten)
        x = self.fc1(x)
        return x

    def forward(self, x):
        return self.forward_without_bn(x)

def main(test_to_run, net, trainset, testset):
    print("CUDA is available:", torch.cuda.is_available())
    # torch.backends.cudnn.deterministic = True
    #n_classes = 10

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
        #transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize, ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch, shuffle=True, num_workers=2)

    #testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
        #transforms.ToTensor(), normalize, ]))
    testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch, shuffle=False, num_workers=2)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #net = NetQ()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)

    # register_linear_layer(net.conv2, "conv2")
    sys.stdout = Logger()

    NumShowInter = 100
    NumEpoch = 300
    IterCounter = -1
    # https://github.com/chengyangfu/pytorch-vgg-cifar10
    training_start = time.time()
    for epoch in range(NumEpoch):  # loop over the dataset multiple times

        #scheduler.step()
        lr = schedule(epoch)
        adjust_learning_rate(optimizer, lr)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            global training_state
            training_state = Config.training
            IterCounter += 1
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % NumShowInter == NumShowInter - 1 or (epoch == 0 and i == 0):  # print every 2000 mini-batches
                elapsed_time = time.time() - training_start
                print('[%d, %5d, %6d, %6f] loss: %.3f' %
                      (epoch + 1, i + 1, IterCounter, elapsed_time, running_loss / NumShowInter))

                running_loss = 0.032
                correct = 0
                total = 0

                # store_layer_name = 'conv2'
                # store_name = f"quantize_{store_layer_name}_{epoch + 1}_{i + 1}"
                # store_layer(store_layer_name, store_name)
                train_store_configs = store_configs.copy()

                with torch.no_grad():
                    training_state = Config.testing
                    for data in testloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

                model_name_base = test_to_run + "_swalp"
                signific_acc = "%03d"%int((correct / total) * 1000)
                loss_name_base = f"./model/{model_name_base}_{signific_acc}"
                os.makedirs("./model/", exist_ok=True)

                #print("Net's state_dict:")
                #for var_name in net.state_dict():
                #        print(var_name, "\t", net.state_dict()[var_name].size())

                torch.save(net.state_dict(), loss_name_base + "_net.pth")
                # model = TheModelClass(*args, **kwargs)
                # model.load_state_dict(torch.load("./model/vgg_swalp_xxxx_net.pth"))
                # model.eval()
                np.save(loss_name_base+"_exp_configs.npy", train_store_configs)
                # Load
                # read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()
                # print(read_dictionary['hello']) # displays "world"


    print('Finished Training')


if __name__ == "__main__":
    _, _, _, test_to_run = argparser_distributed()
    print("====== New Tests ======")
    print("Test To run:", test_to_run)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if test_to_run in ["vgg", "minionn_maxpool", "minionn_cifar10"]:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize, ]))
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(), normalize, ]))
        if test_to_run == "vgg":
            main(test_to_run, NetQ_vgg(10), trainset, testset)
        if test_to_run == "minionn_maxpool":
            main(test_to_run, NetQ_MiniONN_maxpool(), trainset, testset)
        if test_to_run == "minionn_cifar10":
            main(test_to_run, NetQ_MiniONN(), trainset, testset)
    if test_to_run in ["vgg_cifar100", "res34_cifar100", "vgg16_cifar100", "vgg19_cifar100"]:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize, ]))
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(), normalize, ]))
        if test_to_run == "vgg_cifar100":
            main(test_to_run, NetQ_vgg(100), trainset, testset)
        if test_to_run == "res34_cifar100":
            main(test_to_run, NetQ_res34(100), trainset, testset)
        if test_to_run == "vgg16_cifar100":
            main(test_to_run, NetQ_vgg16(100), trainset, testset)
        if test_to_run == "vgg19_cifar100":
            main(test_to_run, NetQ_vgg19(100), trainset, testset)
    if test_to_run == "vgg_idc":
        trainset = torchvision.datasets.ImageFolder(root='../IDC_regular_ps50_idx5/train', transform=transforms.Compose([
           transforms.Resize(size=(32, 32)), transforms.ToTensor(), normalize, ]))
        testset = torchvision.datasets.ImageFolder(root='../IDC_regular_ps50_idx5/val', transform=transforms.Compose([
           transforms.Resize(size=(32, 32)), transforms.ToTensor(), normalize, ]))
        main(test_to_run, NetQ_vgg(2), trainset, testset)
    if test_to_run == "vgg_idc_case2":
        trainset = torchvision.datasets.ImageFolder(root='../IDC_regular_ps50_idx5/train', transform=transforms.Compose([
           transforms.CenterCrop(32), transforms.ToTensor(), normalize, ]))
        testset = torchvision.datasets.ImageFolder(root='../IDC_regular_ps50_idx5/val', transform=transforms.Compose([
           transforms.CenterCrop(32), transforms.ToTensor(), normalize, ]))
        main(test_to_run, NetQ_vgg(2), trainset, testset)
