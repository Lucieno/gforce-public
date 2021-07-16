import os
import re
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

from comm import torch_sync, TrafficRecord
from config import Config
from logger_utils import Logger
from model_reading import get_net_config_name, get_hooking_lst
from modulus_net import shift_by_exp, shift_by_exp_plain, ModulusNet, ModulusNet_vgg16, ModulusNet_MiniONN
from secure_layers import InputSecureLayer, Conv2dSecureLayer, Maxpool2x2SecureLayer, ReluSecureLayer, \
    OutputSecureLayer, FlattenSecureLayer, FcSecureLayer, TruncSecureLayer, Avgpool2x2SecureLayer
from secure_neural_network import SecureNnFramework
from timer_utils import NamedTimerInstance
from torch_utils import argparser_distributed, marshal_funcs, compare_expected_actual, pmod, nmod, warming_up_cuda, \
    MetaTruncRandomGenerator


def generate_secure_vgg(num_class):
    input_img_hw = 32
    input_channel = 3
    input_shape = [input_channel, input_img_hw, input_img_hw]

    input_layer = InputSecureLayer(input_shape, "input_layer")
    conv1 = Conv2dSecureLayer(3, 64, 3, "conv1", padding=1)
    pool1 = Maxpool2x2SecureLayer("pool1")
    relu1 = ReluSecureLayer("relu1")
    trun1 = TruncSecureLayer("trun1")
    conv2 = Conv2dSecureLayer(64, 128, 3, "conv2", padding=1)
    pool2 = Maxpool2x2SecureLayer("pool2")
    relu2 = ReluSecureLayer("relu2")
    trun2 = TruncSecureLayer("trun2")
    conv3 = Conv2dSecureLayer(128, 256, 3, "conv3", padding=1)
    relu3 = ReluSecureLayer("relu3")
    trun3 = TruncSecureLayer("trun3")
    conv4 = Conv2dSecureLayer(256, 256, 3, "conv4", padding=1)
    pool3 = Maxpool2x2SecureLayer("pool3")
    relu4 = ReluSecureLayer("relu4")
    trun4 = TruncSecureLayer("trun4")
    conv5 = Conv2dSecureLayer(256, 512, 3, "conv5", padding=1)
    relu5 = ReluSecureLayer("relu5")
    trun5 = TruncSecureLayer("trun5")
    conv6 = Conv2dSecureLayer(512, 512, 3, "conv6", padding=1)
    pool4 = Maxpool2x2SecureLayer("pool4")
    relu6 = ReluSecureLayer("relu6")
    trun6 = TruncSecureLayer("trun6")
    conv7 = Conv2dSecureLayer(512, 512, 3, "conv7", padding=1)
    relu7 = ReluSecureLayer("relu7")
    trun7 = TruncSecureLayer("trun7")
    conv8 = Conv2dSecureLayer(512, 512, 3, "conv8", padding=1)
    relu8 = ReluSecureLayer("relu8")
    trun8 = TruncSecureLayer("trun8")
    pool5 = Maxpool2x2SecureLayer("pool5")
    flatten = FlattenSecureLayer("flatten")
    fc1 = FcSecureLayer(512, "fc1")
    relu_fc_1 = ReluSecureLayer("relu_fc_1")
    trun_fc_1 = TruncSecureLayer("trun_fc_1")
    fc2 = FcSecureLayer(512, "fc2")
    relu_fc_2 = ReluSecureLayer("relu_fc_2")
    trun_fc_2 = TruncSecureLayer("trun_fc_2")
    fc3 = FcSecureLayer(num_class, "fc3")
    output_layer = OutputSecureLayer("output_layer")
    layers = [input_layer,
              conv1, pool1, relu1, trun1,
              conv2, pool2, relu2, trun2,
              conv3, relu3, trun3,
              conv4, pool3, relu4, trun4,
              conv5, relu5, trun5,
              conv6, relu6, pool4, trun6,
              conv7, relu7, trun7,
              conv8, relu8, pool5, trun8,
              flatten,
              fc1, relu_fc_1, trun_fc_1,
              fc2, relu_fc_2, trun_fc_2,
              fc3,
              output_layer]

    secure_nn = SecureNnFramework("vgg")
    secure_nn.load_layers(layers)

    return secure_nn

def generate_secure_vgg16(num_class):
    input_img_hw = 32
    input_channel = 3
    input_shape = [input_channel, input_img_hw, input_img_hw]

    input_layer = InputSecureLayer(input_shape, "input_layer")
    conv1 = Conv2dSecureLayer(3, 64, 3, "conv1", padding=1)    
    relu1 = ReluSecureLayer("relu1")
    trun1 = TruncSecureLayer("trun1")
    conv2 = Conv2dSecureLayer(64, 64, 3, "conv2", padding=1)
    relu2 = ReluSecureLayer("relu2")
    trun2 = TruncSecureLayer("trun2")
    pool1 = Maxpool2x2SecureLayer("pool1")

    conv3 = Conv2dSecureLayer(64, 128, 3, "conv3", padding=1)
    relu3 = ReluSecureLayer("relu3")
    trun3 = TruncSecureLayer("trun3")
    conv4 = Conv2dSecureLayer(128, 128, 3, "conv4", padding=1)
    pool2 = Maxpool2x2SecureLayer("pool2")
    relu4 = ReluSecureLayer("relu4")
    trun4 = TruncSecureLayer("trun4")

    conv5 = Conv2dSecureLayer(128, 256, 3, "conv5", padding=1)
    relu5 = ReluSecureLayer("relu5")
    trun5 = TruncSecureLayer("trun5")
    conv6 = Conv2dSecureLayer(256, 256, 3, "conv6", padding=1)    
    relu6 = ReluSecureLayer("relu6")
    trun6 = TruncSecureLayer("trun6")
    conv7 = Conv2dSecureLayer(256, 256, 3, "conv7", padding=1)    
    relu7 = ReluSecureLayer("relu7")
    trun7 = TruncSecureLayer("trun7")
    pool3 = Maxpool2x2SecureLayer("pool3")

    conv8 = Conv2dSecureLayer(256, 512, 3, "conv8", padding=1)    
    relu8 = ReluSecureLayer("relu8")
    trun8 = TruncSecureLayer("trun8")
    conv9 = Conv2dSecureLayer(512, 512, 3, "conv9", padding=1)
    relu9 = ReluSecureLayer("relu9")
    trun9 = TruncSecureLayer("trun9")
    conv10 = Conv2dSecureLayer(512, 512, 3, "conv10", padding=1)
    pool4 = Maxpool2x2SecureLayer("pool4")
    relu10 = ReluSecureLayer("relu10")
    trun10 = TruncSecureLayer("trun10")

    conv11 = Conv2dSecureLayer(512, 512, 3, "conv11", padding=1)
    relu11 = ReluSecureLayer("relu11")
    trun11 = TruncSecureLayer("trun11")
    conv12 = Conv2dSecureLayer(512, 512, 3, "conv12", padding=1)
    relu12 = ReluSecureLayer("relu12")
    trun12 = TruncSecureLayer("trun12")
    conv13 = Conv2dSecureLayer(512, 512, 3, "conv13", padding=1)
    relu13 = ReluSecureLayer("relu13")
    trun13 = TruncSecureLayer("trun13")
    pool5 = Maxpool2x2SecureLayer("pool5")

    flatten = FlattenSecureLayer("flatten")
    fc1 = FcSecureLayer(512, "fc1")
    relu_fc_1 = ReluSecureLayer("relu_fc_1")
    trun_fc_1 = TruncSecureLayer("trun_fc_1")
    fc2 = FcSecureLayer(512, "fc2")
    relu_fc_2 = ReluSecureLayer("relu_fc_2")
    trun_fc_2 = TruncSecureLayer("trun_fc_2")
    fc3 = FcSecureLayer(num_class, "fc3")
    output_layer = OutputSecureLayer("output_layer")
    layers = [input_layer,
              conv1, relu1, trun1,
              conv2, pool1, relu2, trun2,
              conv3, relu3, trun3,
              conv4, pool2, relu4, trun4,
              conv5, relu5, trun5,
              conv6, relu6, trun6,
              conv7, pool3, relu7, trun7,
              conv8, relu8, trun8,
              conv9, relu9, trun9,
              conv10, pool4, relu10, trun10,
              conv11, relu11, trun11,
              conv12, relu12, trun12,
              conv13, pool5, relu13, trun13,
              flatten,
              fc1, relu_fc_1, trun_fc_1,
              fc2, relu_fc_2, trun_fc_2,
              fc3,
              output_layer]

    secure_nn = SecureNnFramework("vgg16")
    secure_nn.load_layers(layers)

    return secure_nn

def generate_secure_minionn(pooling_type_name):
    input_img_hw = 32
    input_channel = 3
    input_shape = [input_channel, input_img_hw, input_img_hw]

    if pooling_type_name == "maxpool":
        pool = Maxpool2x2SecureLayer
    elif pooling_type_name == "avgpool":
        pool = Avgpool2x2SecureLayer
    else:
        raise Exception(f"Unknown Pooling Type Name: {pooling_type_name}")


    input_layer = InputSecureLayer(input_shape, "input_layer")
    conv1 = Conv2dSecureLayer(3, 64, 3, "conv1", padding=1)
    relu1 = ReluSecureLayer("relu1")
    trun1 = TruncSecureLayer("trun1")
    conv2 = Conv2dSecureLayer(64, 64, 3, "conv2", padding=1)
    relu2 = ReluSecureLayer("relu2")
    pool1 = pool("pool1")
    trun2 = TruncSecureLayer("trun2")
    conv3 = Conv2dSecureLayer(64, 64, 3, "conv3", padding=1)
    relu3 = ReluSecureLayer("relu3")
    trun3 = TruncSecureLayer("trun3")
    conv4 = Conv2dSecureLayer(64, 64, 3, "conv4", padding=1)
    pool2 = pool("pool2")
    relu4 = ReluSecureLayer("relu4")
    trun4 = TruncSecureLayer("trun4")
    conv5 = Conv2dSecureLayer(64, 64, 3, "conv5", padding=1)
    relu5 = ReluSecureLayer("relu5")
    trun5 = TruncSecureLayer("trun5")
    conv6 = Conv2dSecureLayer(64, 64, 3, "conv6", padding=1)
    relu6 = ReluSecureLayer("relu6")
    trun6 = TruncSecureLayer("trun6")
    conv7 = Conv2dSecureLayer(64, 16, 3, "conv7", padding=1)
    relu8 = ReluSecureLayer("relu8")
    flatten = FlattenSecureLayer("flatten")
    trun7 = TruncSecureLayer("trun7")
    fc1 = FcSecureLayer(10, "fc1")
    # fc1 = FcSecureLayer(2, "fc1")
    output_layer = OutputSecureLayer("output_layer")
    layers = [input_layer,
              conv1, relu1, trun1,
              conv2, pool1, relu2, trun2,
              conv3, relu3, trun3,
              conv4, pool2, relu4, trun4,
              conv5, relu5, trun5,
              conv6, relu6, trun6,
              conv7, relu8, trun7,
              flatten,
              fc1,
              output_layer]

    secure_nn = SecureNnFramework("minionn_avgpool")
    secure_nn.load_layers(layers)

    return secure_nn


def load_trunc_params(secure_nn: SecureNnFramework, store_configs: dict):
    linear_layers = []
    trunc_layer_relation = []
    avgpool_indices = []
    cur_linear_layer_index = -1
    for layer in secure_nn.layers:
        if isinstance(layer, Conv2dSecureLayer) or isinstance(layer, FcSecureLayer):
            linear_layers.append(layer)
            cur_linear_layer_index += 1
        elif isinstance(layer, Avgpool2x2SecureLayer):
            avgpool_indices.append(cur_linear_layer_index)
        elif isinstance(layer, TruncSecureLayer):
            trunc_layer_relation.append((layer, cur_linear_layer_index))

    def calc_shift_pow(prev_layer_name, next_layer_name):
        bit, ebit = 8, 8

        input_exp, _ = store_configs[prev_layer_name + "ForwardX"]
        weight_exp, _ = store_configs[prev_layer_name + "ForwardY"]
        next_input_exp, _ = store_configs[next_layer_name + "ForwardX"]

        exp = weight_exp + input_exp - next_input_exp - bit + 2

        return -exp

    for trunc_layer, linear_layer_index in trunc_layer_relation:
        prev_layer_name = linear_layers[linear_layer_index].name
        next_layer_name = linear_layers[linear_layer_index+1].name
        pow = calc_shift_pow(prev_layer_name, next_layer_name)
        if linear_layer_index in avgpool_indices:
            pow += 2
        trunc_layer.set_div_to_pow(pow)


def load_weight_params(secure_nn: SecureNnFramework, store_configs: dict, net_state: dict):
    linear_layers = []
    for layer in secure_nn.layers:
        if isinstance(layer, Conv2dSecureLayer) or isinstance(layer, FcSecureLayer):
            linear_layers.append(layer)
    bits = 8
    # print("net_state.keys()", net_state.keys())
    for layer in linear_layers:
        input_exp, _ = store_configs[layer.name + "ForwardX"]
        weight_exp, _ = store_configs[layer.name + "ForwardY"]
        w_exp = -weight_exp + (bits - 2)
        exp = -weight_exp + (bits - 2) -input_exp + (bits - 2)
        weight_f = net_state[layer.name + ".weight"].double()
        bias_name = layer.name + ".bias"
        if bias_name in net_state:
            bias_f = net_state[bias_name].double()
            shifted_bias = shift_by_exp_plain(bias_f, exp)
        else:
            shifted_bias = None
        layer.load_weight(shift_by_exp_plain(weight_f, w_exp), shifted_bias)


def data_shift(data, config):
    bits = 8
    input_exp, _ = config
    exp = -input_exp + (bits - 2)
    res = shift_by_exp_plain(data, exp)
    res.clamp_(-2 ** (bits - 1), 2 ** (bits - 1) - 1)

    return res


def get_file_highest_acc(model_name_base):
    accs = []
    path = "./model/"
    for root, dirs, files in os.walk(path):
        for name in files:
            pattern = model_name_base.replace("_", "\_") + "\_(\d+)" + "\_exp\_configs[.]npy"
            print(name, pattern)
            acc = re.findall(pattern, name)
            if len(acc) > 0:
                accs.append(int(acc[0]))
    return max(accs)


def secure_vgg(input_sid, master_addr, master_port, model_name_base="vgg_swalp"):
    test_name = "secure inference"
    print(f"\nTest for {test_name}: Start")

    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    batch_size = 1

    net_state_name, config_name = get_net_config_name(model_name_base)
    print(f"net_state going to load: {net_state_name}")
    print(f"store_configs going to load: {config_name}")

    store_configs = np.load(config_name, allow_pickle="TRUE").item()

    def get_plain_net():
        if model_name_base == "vgg_swalp":
            net = ModulusNet(store_configs)
        elif model_name_base == "vgg_idc_swalp":
            net = ModulusNet(store_configs, 2)
        elif model_name_base == "vgg_cifar100":
            net = ModulusNet(store_configs, 100)
        elif model_name_base == "vgg16_cifar100":
            net = ModulusNet_vgg16(store_configs, 100)
        elif model_name_base == "vgg16_cifar10":
            net = ModulusNet_vgg16(store_configs, 10)
        elif model_name_base == "minionn_maxpool":
            net = ModulusNet_MiniONN(store_configs)
        else:
            raise Exception("Unknown: {model_name_base}")
        net_state = torch.load(net_state_name)
        device = torch.device("cuda:0")
        net.load_weight_bias(net_state)
        net.to(device)
        return net

    def get_secure_nn():
        if model_name_base == "vgg_swalp":
            return generate_secure_vgg(10)
        elif model_name_base == "vgg_idc_swalp":
            return generate_secure_vgg(2)
        elif model_name_base == "vgg_cifar100":
            return generate_secure_vgg(100)
        elif model_name_base == "vgg16_cifar100":
            return generate_secure_vgg16(100)
        elif model_name_base == "vgg16_cifar10":
            return generate_secure_vgg16(10)
        elif model_name_base == "minionn_cifar10":
            return generate_secure_minionn("avgpool")
        elif model_name_base == "minionn_maxpool":
            return generate_secure_minionn("maxpool")
        else:
            raise Exception("Unknown: {model_name_base}")

    def check_correctness(self, input_img, output, modulus):
        plain_net = get_plain_net()
        expected = plain_net(input_img.reshape([1] + list(input_img.shape)).cuda())
        expected = nmod(expected.reshape(expected.shape[1:]), modulus)
        actual = nmod(output, modulus).cuda()
        print("expected", expected)
        print("actual", actual)
        compare_expected_actual(expected, actual, name="secure_vgg", get_relative=True)

        _, expected_max = torch.max(expected, 0)
        _, actual_max = torch.max(actual, 0)
        print(f"expected_max: {expected_max}, actual_max: {actual_max}, Match: {expected_max == actual_max}")

    # check_correctness(None, torch.zeros([3, 32, 32]) + 1, torch.zeros(10), secure_nn.q_23)

    def test_server():
        rank = Config.server_rank
        sys.stdout = Logger()
        traffic_record = TrafficRecord()
        secure_nn = get_secure_nn()
        secure_nn.set_rank(rank).init_communication(master_address=master_addr, master_port=master_port)
        warming_up_cuda()
        secure_nn.fhe_builder_sync()
        load_trunc_params(secure_nn, store_configs)

        net_state = torch.load(net_state_name)
        load_weight_params(secure_nn, store_configs, net_state)

        meta_rg = MetaTruncRandomGenerator()
        meta_rg.reset_seed()

        with NamedTimerInstance("Server Offline"):
            secure_nn.offline()
            torch_sync()
        traffic_record.reset("server-offline")

        with NamedTimerInstance("Server Online"):
            secure_nn.online()
            torch_sync()
        traffic_record.reset("server-online")

        secure_nn.check_correctness(check_correctness)
        secure_nn.check_layers(get_plain_net, get_hooking_lst(model_name_base))
        secure_nn.end_communication()

    def test_client():
        rank = Config.client_rank
        sys.stdout = Logger()
        traffic_record = TrafficRecord()
        secure_nn = get_secure_nn()
        secure_nn.set_rank(rank).init_communication(master_address=master_addr, master_port=master_port)
        warming_up_cuda()
        secure_nn.fhe_builder_sync()

        load_trunc_params(secure_nn, store_configs)

        def input_shift(data):
            first_layer_name = "conv1"
            return data_shift(data, store_configs[first_layer_name + "ForwardX"])

        def testset():
            if model_name_base in ["vgg16_cifar100"]:
                return torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        input_shift
                        ]))
            elif model_name_base in ["vgg16_cifar10", "minionn_maxpool"]:
                return torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        input_shift
                        ]))

        testloader = torch.utils.data.DataLoader(testset(), batch_size=batch_size, shuffle=True, num_workers=2)

        data_iter = iter(testloader)
        image, truth = next(data_iter)
        image = image.reshape(secure_nn.get_input_shape())
        secure_nn.fill_input(image)

        with NamedTimerInstance("Client Offline"):
            secure_nn.offline()
            torch_sync()
        traffic_record.reset("client-offline")

        with NamedTimerInstance("Client Online"):
            secure_nn.online()
            torch_sync()
        traffic_record.reset("client-online")

        secure_nn.check_correctness(check_correctness, truth=truth)
        secure_nn.check_layers(get_plain_net, get_hooking_lst(model_name_base))
        secure_nn.end_communication()

    if input_sid == Config.server_rank:
        # test_server()
        marshal_funcs([test_server])
    elif input_sid == Config.client_rank:
        # test_client()
        marshal_funcs([test_client])
    elif input_sid == Config.both_rank:
        marshal_funcs([test_server, test_client])
    else:
        raise Exception(f"Unknown input_sid: {input_sid}")

    print(f"\nTest for {test_name}: End")

if __name__ == "__main__":
    input_sid, master_addr, master_port, test_to_run = argparser_distributed()
    sys.stdout = Logger()

    print("====== New Tests ======")

    num_repeat = 10

    if test_to_run in ["all", "vgg"]:
        model_name_base = "vgg16_cifar10"
    elif test_to_run in ["vgg_idc"]:
        model_name_base = "vgg_idc_swalp"
    elif test_to_run in ["vgg_cifar100"]:
        model_name_base = "vgg_cifar100"
    elif test_to_run in ["vgg16_cifar100"]:
        model_name_base = "vgg16_cifar100"
    elif test_to_run in ["vgg16_cifar10"]:
        model_name_base = "vgg16_cifar10"
    elif test_to_run in ["minionn_avgpool"]:
        model_name_base = "minionn_cifar10_swalp"
    elif test_to_run in ["minionn_maxpool"]:
        model_name_base = "minionn_maxpool"

    for _ in range(num_repeat):
        secure_vgg(input_sid, master_addr, master_port, model_name_base)
