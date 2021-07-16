import argparse
import sys
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.multiprocessing import Process


from comm import NamedBase, end_communicate, torch_sync, init_communicate, CommFheBuilder, BlobTorch, CommBase, \
    TrafficRecord
from config import Config
from fhe import FheBuilder
from logger_utils import Logger
from secure_layers import SecureLayerContext, SecureNeuralNetwork, ContextRankBase, SecureLayerBase, InputSecureLayer, \
    Conv2dSecureLayer, Maxpool2x2SecureLayer, ReluSecureLayer, FlattenSecureLayer, FcSecureLayer, OutputSecureLayer
from timer_utils import NamedTimerInstance
from torch_utils import get_prod, gen_unirand_int_grain, pmod, compare_expected_actual, nmod, argparser_distributed, \
    warming_up_cuda, MetaTruncRandomGenerator


class SecureNnFramework(NamedBase, ContextRankBase):
    class_name = "SecureNnFramework"
    layers: List[SecureLayerBase]
    layer_dict: Dict[str, SecureLayerBase]
    input_img: torch.Tensor
    output_res: torch.Tensor
    comm_base: CommBase

    # def __init__(self, name, data_bit=5, work_bit=18, q_16=12289, q_23=7340033, input_hw=32, input_channel=3):
    def __init__(self, name, data_bit=5, work_bit=18,
                 n_16=Config.n_16, n_23=Config.n_23, q_16=Config.q_16, q_23=Config.q_23,
                 input_hw=32, input_channel=3):

        super().__init__(name)
        self.data_bit = data_bit
        self.work_bit = work_bit
        self.q_16 = q_16
        self.q_23 = q_23
        self.input_hw = input_hw
        self.input_channel = input_channel

        self.data_range = 2 ** data_bit
        self.fhe_builder_16 = FheBuilder(q_16, n_16)
        self.fhe_builder_23 = FheBuilder(q_23, n_23)

        self.context = SecureLayerContext(work_bit, data_bit, q_16, q_23, self.fhe_builder_16, self.fhe_builder_23,
                                          self.sub_name("context"))

        self.secure_nn_core = SecureNeuralNetwork(self.sub_name("secure_nn"))

        self.layer_dict = {}

    def generate_random_data(self, shape):
        return gen_unirand_int_grain(-self.data_range//2 + 1, self.data_range//2, get_prod(shape)).reshape(shape)

    def load_layers(self, layers):
        self.layers = layers
        for layer in layers:
            self.layer_dict[layer.name] = layer

        self.secure_nn_core.load_layers(layers)
        self.secure_nn_core.load_context(self.context)
        return self

    def get_input_shape(self):
        return self.secure_nn_core.input_layer.get_output_shape()

    def get_output_shape(self):
        return self.secure_nn_core.output_layer.get_output_shape()

    def set_rank(self, rank):
        self.rank = rank
        self.context.set_rank(rank)
        return self

    def init_communication(self, **kwargs):
        init_communicate(self.rank, **kwargs)
        self.comm_base = CommBase(self.rank, self.sub_name("comm_base"))
        return self

    def fhe_builder_sync(self):
        comm_fhe_16 = CommFheBuilder(self.rank, self.fhe_builder_16, self.sub_name("fhe_builder_16"))
        comm_fhe_23 = CommFheBuilder(self.rank, self.fhe_builder_23, self.sub_name("fhe_builder_23"))
        torch_sync()
        if self.is_server():
            comm_fhe_16.recv_public_key()
            comm_fhe_23.recv_public_key()
            comm_fhe_16.wait_and_build_public_key()
            comm_fhe_23.wait_and_build_public_key()
        elif self.is_client():
            self.fhe_builder_16.generate_keys()
            self.fhe_builder_23.generate_keys()
            comm_fhe_16.send_public_key()
            comm_fhe_23.send_public_key()
        torch_sync()
        return self

    def fill_random_weight(self):
        assert(self.is_server())
        for layer in self.layers:
            if not layer.has_weight: continue
            layer.load_weight(self.generate_random_data(layer.weight_shape))
        return self

    def fill_input(self, input_img):
        assert(self.is_client())
        self.input_img = input_img
        self.secure_nn_core.feed_input(input_img)
        return self

    def fill_random_input(self):
        assert(self.is_client())
        self.fill_input(self.generate_random_data(self.get_input_shape()))
        return self

    def offline(self):
        self.secure_nn_core.offline()
        return self

    def online(self):
        self.secure_nn_core.online()
        return self

    def check_layers(self, get_plain_net_func, all_layer_pair):
        secure_input_layer = self.layers[0]
        secure_input_layer.reconstructed_to_server(self.comm_base, self.q_23)

        for secure_layer_name, plain_layer_name in all_layer_pair:
            secure_layer = self.layer_dict[secure_layer_name]
            secure_layer.reconstructed_to_server(self.comm_base, self.q_23)

        if self.is_server():
            plain_net = get_plain_net_func()
            plain_output_dict = {}

            def hook_generator(name):
                def hook(module, input, output):
                    plain_output_dict[name] = output.data.detach().clone()
                return hook

            for secure_layer_name, plain_layer_name in all_layer_pair:
                # plain_output_dict[plain_layer_name] = torch.zeros()

                plain_layer = getattr(plain_net, plain_layer_name)
                plain_layer.register_forward_hook(hook_generator(plain_layer_name))

            input_img = secure_input_layer.get_reconstructed_output()
            input_img = input_img.reshape([1] + list(input_img.shape)).cuda()

            meta_rg = MetaTruncRandomGenerator()
            meta_rg.reset_rg("plain")
            plain_output = plain_net(input_img)

            for secure_layer_name, plain_layer_name in all_layer_pair:
                print("secure_layer_name, plain_layer_name: %s, %s"%(secure_layer_name, plain_layer_name))
                plain_output = plain_output_dict[plain_layer_name]
                secure_layer = self.layer_dict[secure_layer_name]
                secure_output = secure_layer.get_reconstructed_output()
                compare_expected_actual(plain_output, secure_output,
                                        name=f"compare secure-plain: {secure_layer_name}, {plain_layer_name}", get_relative=True)
                # print("secure", secure_output.shape, secure_output)
                # print("plain", plain_output.shape, plain_output)

        return self

    def check_correctness(self, verify_func, is_argmax=False, truth=None):
        blob_input_img = BlobTorch(self.get_input_shape(), torch.float, self.comm_base, "input_img")
        blob_actual_output = BlobTorch(self.get_output_shape(), torch.float, self.comm_base, "actual_output")
        blob_truth = BlobTorch(1, torch.float, self.comm_base, "truth")

        if self.is_server():
            blob_input_img.prepare_recv()
            blob_actual_output.prepare_recv()
            blob_truth.prepare_recv()
            torch_sync()
            input_img = blob_input_img.get_recv()
            actual_output = blob_actual_output.get_recv()
            truth = int(blob_truth.get_recv().item())
            verify_func(self, input_img, actual_output, self.q_23)

            actual_output = nmod(actual_output, self.q_23).cuda()
            _, actual_max = torch.max(actual_output, 0)
            print(f"truth: {truth}, actual: {actual_max}, MatchTruth: {truth == actual_max}")

        if self.is_client():
            torch_sync()
            actual_output = self.secure_nn_core.get_argmax_output() if is_argmax else self.secure_nn_core.get_output()
            blob_input_img.send(self.input_img)
            blob_actual_output.send(actual_output)
            blob_truth.send(torch.tensor(truth))

        return self

    def end_communication(self):
        end_communicate()
        return self


def run_secure_nn_server_with_random_data(secure_nn, check_correctness, master_address, master_port):
    rank = Config.server_rank
    traffic_record = TrafficRecord()
    secure_nn.set_rank(rank).init_communication(master_address=master_address, master_port=master_port)
    warming_up_cuda()
    secure_nn.fhe_builder_sync()
    secure_nn.fill_random_weight()

    with NamedTimerInstance("Server Offline"):
        secure_nn.offline()
        torch_sync()
    traffic_record.reset("server-offline")

    with NamedTimerInstance("Server Online"):
        secure_nn.online()
        torch_sync()
    traffic_record.reset("server-online")

    secure_nn.check_correctness(check_correctness).end_communication()


def run_secure_nn_client_with_random_data(secure_nn, check_correctness, master_address, master_port):
    rank = Config.client_rank
    traffic_record = TrafficRecord()
    secure_nn.set_rank(rank).init_communication(master_address=master_address, master_port=master_port)
    warming_up_cuda()
    secure_nn.fhe_builder_sync().fill_random_input()

    with NamedTimerInstance("Client Offline"):
        secure_nn.offline()
        torch_sync()
    traffic_record.reset("client-offline")

    with NamedTimerInstance("Client Online"):
        secure_nn.online()
        torch_sync()
    traffic_record.reset("client-online")

    secure_nn.check_correctness(check_correctness).end_communication()


def generate_relu_only_nn():
    num_elem = 2 ** 17
    input_img_hw = 16
    input_channel = num_elem // (input_img_hw ** 2)
    assert(input_img_hw * input_img_hw * input_channel == num_elem)
    input_shape = [input_channel, input_img_hw, input_img_hw]

    input_layer = InputSecureLayer(input_shape, "input_layer")
    relu1 = ReluSecureLayer("relu1")
    output_layer = OutputSecureLayer("output_layer")
    layers = [input_layer, relu1, output_layer]

    secure_nn = SecureNnFramework("relu_only_nn")
    secure_nn.load_layers(layers)

    return secure_nn


def correctness_relu_only_nn(self, input_img, output, modulus):
    x = input_img.cuda().double()
    x = x.reshape([1] + list(x.shape))
    x = pmod(F.relu(nmod(x, modulus)), modulus)

    expected = x
    actual = pmod(output, modulus)
    if len(expected.shape) == 4 and expected.shape[0] == 1:
        expected = expected.reshape(expected.shape[1:])
    compare_expected_actual(expected, actual, name="relu_only_nn", get_relative=True)


def generate_maxpool2x2():
    num_elem = 2 ** 17
    input_img_hw = 16
    input_channel = num_elem // (input_img_hw ** 2)
    assert(input_img_hw * input_img_hw * input_channel == num_elem)
    input_shape = [input_channel, input_img_hw, input_img_hw]

    input_layer = InputSecureLayer(input_shape, "input_layer")
    pool1 = Maxpool2x2SecureLayer("pool1")
    output_layer = OutputSecureLayer("output_layer")
    layers = [input_layer, pool1, output_layer]

    secure_nn = SecureNnFramework("maxpool2x2")
    secure_nn.load_layers(layers)

    return secure_nn


def correctness_maxpool2x2(self, input_img, output, modulus):
    torch_pool1 = torch.nn.MaxPool2d(2)

    x = input_img.cuda().double()
    x = x.reshape([1] + list(x.shape))
    x = pmod(torch_pool1(nmod(x, modulus)), modulus)

    expected = x
    actual = pmod(output, modulus)
    if len(expected.shape) == 4 and expected.shape[0] == 1:
        expected = expected.reshape(expected.shape[1:])
    compare_expected_actual(expected, actual, name="maxpool2x2", get_relative=True)


def generate_conv2d():
    input_img_hw = 16
    input_channel = 128
    input_shape = [input_channel, input_img_hw, input_img_hw]

    input_layer = InputSecureLayer(input_shape, "input_layer")
    conv1 = Conv2dSecureLayer(128, 128, 3, "conv1", padding=1)
    output_layer = OutputSecureLayer("output_layer")
    layers = [input_layer, conv1, output_layer]

    secure_nn = SecureNnFramework("conv2d")
    secure_nn.load_layers(layers)

    return secure_nn


def correctness_conv2d(self, input_img, output, modulus):
    x = input_img.cuda().double()
    x = x.reshape([1] + list(x.shape))
    x = pmod(F.conv2d(x, self.layers[1].weight.cuda().double(), padding=1), modulus)

    expected = x
    actual = pmod(output, modulus)
    if len(expected.shape) == 4 and expected.shape[0] == 1:
        expected = expected.reshape(expected.shape[1:])
    compare_expected_actual(expected, actual, name="conv2d", get_relative=True)


def generate_fc():
    num_input_unit = 1024
    num_output_unit = 128
    input_shape = [num_input_unit]

    input_layer = InputSecureLayer(input_shape, "input_layer")
    fc1 = FcSecureLayer(num_output_unit, "fc1")
    output_layer = OutputSecureLayer("output_layer")
    layers = [input_layer, fc1, output_layer]

    secure_nn = SecureNnFramework("fc")
    secure_nn.load_layers(layers)

    return secure_nn


def correctness_fc(self, input_img, output, modulus):
    x = input_img.cuda().double()
    x = pmod(torch.mm(x.view(1, -1), self.layers[1].weight.cuda().double().t()).view(-1), modulus)

    expected = x
    actual = pmod(output, modulus)
    if len(expected.shape) == 4 and expected.shape[0] == 1:
        expected = expected.reshape(expected.shape[1:])
    compare_expected_actual(expected, actual, name="fc", get_relative=True)


def generate_small_nn():
    input_img_hw = 16
    input_channel = 3
    input_shape = [input_channel, input_img_hw, input_img_hw]

    input_layer = InputSecureLayer(input_shape, "input_layer")
    conv1 = Conv2dSecureLayer(3, 5, 3, "conv1", padding=1)
    relu1 = ReluSecureLayer("relu1")
    pool1 = Maxpool2x2SecureLayer("pool1")
    conv2 = Conv2dSecureLayer(5, 10, 3, "conv2", padding=1)
    flatten = FlattenSecureLayer("flatten")
    fc1 = FcSecureLayer(512, "fc1")
    output_layer = OutputSecureLayer("output_layer")
    layers = [input_layer, conv1, pool1, relu1, conv2, flatten, fc1, output_layer]

    secure_nn = SecureNnFramework("small_nn")
    secure_nn.load_layers(layers)

    return secure_nn


def correctness_small_nn(self, input_img, output, modulus):
    torch_pool1 = torch.nn.MaxPool2d(2)

    x = input_img.cuda().double()
    x = x.reshape([1] + list(x.shape))
    x = pmod(F.conv2d(x, self.layers[1].weight.cuda().double(), padding=1), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(torch_pool1(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[4].weight.cuda().double(), padding=1), modulus)
    x = x.view(-1)
    x = pmod(torch.mm(x.view(1, -1), self.layers[6].weight.cuda().double().t()).view(-1), modulus)

    expected = x
    actual = pmod(output, modulus)
    if len(expected.shape) == 4 and expected.shape[0] == 1:
        expected = expected.reshape(expected.shape[1:])
    compare_expected_actual(expected, actual, name="small_nn", get_relative=True)

def generate_minionn_cifar10():
    input_img_hw = 32
    input_channel = 3
    input_shape = [input_channel, input_img_hw, input_img_hw]

    input_layer = InputSecureLayer(input_shape, "input_layer")
    conv1 = Conv2dSecureLayer(3, 64, 3, "conv1", padding=1)
    relu1 = ReluSecureLayer("relu1")
    conv2 = Conv2dSecureLayer(64, 64, 3, "conv2", padding=1)
    pool1 = Maxpool2x2SecureLayer("pool1")
    relu2 = ReluSecureLayer("relu2")
    conv3 = Conv2dSecureLayer(64, 64, 3, "conv3", padding=1)
    relu3 = ReluSecureLayer("relu3")
    conv4 = Conv2dSecureLayer(64, 64, 3, "conv4", padding=1)
    pool2 = Maxpool2x2SecureLayer("pool2")
    relu4 = ReluSecureLayer("relu4")
    conv5 = Conv2dSecureLayer(64, 64, 3, "conv5", padding=1)
    relu5 = ReluSecureLayer("relu5")
    conv6 = Conv2dSecureLayer(64, 64, 3, "conv6", padding=1)
    relu6 = ReluSecureLayer("relu6")
    conv7 = Conv2dSecureLayer(64, 16, 3, "conv7", padding=1)
    relu8 = ReluSecureLayer("relu8")
    flatten = FlattenSecureLayer("flatten")
    fc1 = FcSecureLayer(10, "fc1")
    output_layer = OutputSecureLayer("output_layer")
    layers = [input_layer,
              conv1, relu1,
              conv2, pool1, relu2, conv3, relu3,
              conv4, pool2, relu4, conv5, relu5, conv6, relu6, conv7, relu8,
              flatten, fc1,
              output_layer]

    secure_nn = SecureNnFramework("minionn_cifar10")
    secure_nn.load_layers(layers)

    return secure_nn


def correctness_minionn_cifar10(self, input_img, output, modulus):
    torch_pool1 = torch.nn.MaxPool2d(2)
    torch_pool2 = torch.nn.MaxPool2d(2)

    x = input_img.cuda().double()
    x = x.reshape([1] + list(x.shape))
    x = pmod(F.conv2d(x, self.layers[1].weight.cuda().double(), padding=1), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[3].weight.cuda().double(), padding=1), modulus)
    x = pmod(torch_pool1(nmod(x, modulus)), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[6].weight.cuda().double(), padding=1), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[8].weight.cuda().double(), padding=1), modulus)
    x = pmod(torch_pool2(nmod(x, modulus)), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[11].weight.cuda().double(), padding=1), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[13].weight.cuda().double(), padding=1), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[15].weight.cuda().double(), padding=1), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = x.view(-1)
    x = pmod(torch.mm(x.view(1, -1), self.layers[18].weight.cuda().double().t()).view(-1), modulus)

    expected = x
    actual = pmod(output, modulus)
    if len(expected.shape) == 4 and expected.shape[0] == 1:
        expected = expected.reshape(expected.shape[1:])
    compare_expected_actual(expected, actual, name="minionn_mnist", get_relative=True)


def generate_minionn_mnist():
    input_img_hw = 28
    input_channel = 1
    input_shape = [input_channel, input_img_hw, input_img_hw]

    input_layer = InputSecureLayer(input_shape, "input_layer")
    conv1 = Conv2dSecureLayer(input_channel, 16, 5, "conv1", padding=0)
    pool1 = Maxpool2x2SecureLayer("pool1")
    relu1 = ReluSecureLayer("relu1")
    conv2 = Conv2dSecureLayer(16, 16, 5, "conv2", padding=0)
    pool2 = Maxpool2x2SecureLayer("pool2")
    relu2 = ReluSecureLayer("relu2")
    flatten = FlattenSecureLayer("flatten")
    fc1 = FcSecureLayer(100, "fc1")
    relu3 = ReluSecureLayer("relu3")
    fc2 = FcSecureLayer(10, "fc2")
    output_layer = OutputSecureLayer("output_layer")
    layers = [input_layer, conv1, pool1, relu1, conv2, pool2, relu2, flatten, fc1, relu3, fc2, output_layer]

    secure_nn = SecureNnFramework("minionn_mnist")
    secure_nn.load_layers(layers)

    return secure_nn


def correctness_minionn_mnist(self, input_img, output, modulus):
    torch_pool1 = torch.nn.MaxPool2d(2)
    torch_pool2 = torch.nn.MaxPool2d(2)

    x = input_img.cuda().double()
    x = x.reshape([1] + list(x.shape))
    x = pmod(F.conv2d(x, self.layers[1].weight.cuda().double(), padding=0), modulus)
    x = pmod(torch_pool1(nmod(x, modulus)), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[4].weight.cuda().double(), padding=0), modulus)
    x = pmod(torch_pool2(nmod(x, modulus)), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = x.view(-1)
    x = pmod(torch.mm(x.view(1, -1), self.layers[8].weight.cuda().double().t()).view(-1), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(torch.mm(x.view(1, -1), self.layers[10].weight.cuda().double().t()).view(-1), modulus)

    expected = x
    actual = pmod(output, modulus)
    if len(expected.shape) == 4 and expected.shape[0] == 1:
        expected = expected.reshape(expected.shape[1:])
    compare_expected_actual(expected, actual, name="minionn_minst", get_relative=True)


def generate_vgg():
    input_img_hw = 32
    input_channel = 3
    input_shape = [input_channel, input_img_hw, input_img_hw]

    input_layer = InputSecureLayer(input_shape, "input_layer")
    conv1 = Conv2dSecureLayer(3, 64, 3, "conv1", padding=1)
    pool1 = Maxpool2x2SecureLayer("pool1")
    relu1 = ReluSecureLayer("relu1")
    conv2 = Conv2dSecureLayer(64, 128, 3, "conv2", padding=1)
    pool2 = Maxpool2x2SecureLayer("pool2")
    relu2 = ReluSecureLayer("relu2")
    conv3 = Conv2dSecureLayer(128, 256, 3, "conv3", padding=1)
    relu3 = ReluSecureLayer("relu3")
    conv4 = Conv2dSecureLayer(256, 256, 3, "conv4", padding=1)
    pool3 = Maxpool2x2SecureLayer("pool3")
    relu4 = ReluSecureLayer("relu4")
    conv5 = Conv2dSecureLayer(256, 512, 3, "conv5", padding=1)
    relu5 = ReluSecureLayer("relu5")
    conv6 = Conv2dSecureLayer(512, 512, 3, "conv6", padding=1)
    relu6 = ReluSecureLayer("relu6")
    pool4 = Maxpool2x2SecureLayer("pool4")
    conv7 = Conv2dSecureLayer(512, 512, 3, "conv7", padding=1)
    relu7 = ReluSecureLayer("relu7")
    conv8 = Conv2dSecureLayer(512, 512, 3, "conv8", padding=1)
    relu8 = ReluSecureLayer("relu8")
    pool5 = Maxpool2x2SecureLayer("pool5")
    flatten = FlattenSecureLayer("flatten")
    fc1 = FcSecureLayer(512, "fc1")
    relu_fc_1 = ReluSecureLayer("relu_fc_1")
    fc2 = FcSecureLayer(512, "fc2")
    relu_fc_2 = ReluSecureLayer("relu_fc_2")
    fc3 = FcSecureLayer(10, "fc3")
    output_layer = OutputSecureLayer("output_layer")
    layers = [input_layer,
              conv1, pool1, relu1, conv2, pool2, relu2,
              conv3, relu3, conv4, pool3, relu4, conv5, relu5, conv6, relu6, pool4,
              conv7, relu7, conv8, relu8, pool5,
              flatten,
              fc1, relu_fc_1, fc2, relu_fc_2, fc3,
              output_layer]

    secure_nn = SecureNnFramework("vgg")
    secure_nn.load_layers(layers)

    return secure_nn


def correctness_vgg(self, input_img, output, modulus):
    return
    torch_pool1 = torch.nn.MaxPool2d(2)
    torch_pool2 = torch.nn.MaxPool2d(2)

    x = input_img.cuda().double()
    x = x.reshape([1] + list(x.shape))
    x = pmod(F.conv2d(x, self.layers[1].weight.cuda().double(), padding=1), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[3].weight.cuda().double(), padding=1), modulus)
    x = pmod(torch_pool1(nmod(x, modulus)), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[6].weight.cuda().double(), padding=1), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[8].weight.cuda().double(), padding=1), modulus)
    x = pmod(torch_pool2(nmod(x, modulus)), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[11].weight.cuda().double(), padding=1), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[13].weight.cuda().double(), padding=1), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = pmod(F.conv2d(x, self.layers[15].weight.cuda().double(), padding=1), modulus)
    x = pmod(F.relu(nmod(x, modulus)), modulus)
    x = x.view(-1)
    x = pmod(torch.mm(x.view(1, -1), self.layers[18].weight.cuda().double().t()).view(-1), modulus)

    expected = x
    actual = pmod(output, modulus)
    if len(expected.shape) == 4 and expected.shape[0] == 1:
        expected = expected.reshape(expected.shape[1:])
    compare_expected_actual(expected, actual, name="minionn_mnist", get_relative=True)


def marshal_secure_nn_parties(party, master_address, master_port, secure_nn, correctness_func):
    test_name = "secure nn"
    print(f"\nTest for {test_name}: Start")

    processes = []

    if party == Config.both_rank:
        p = Process(target=run_secure_nn_server_with_random_data,
                    args=[secure_nn, correctness_func, master_address, master_port])
        p.start()
        processes.append(p)
        p = Process(target=run_secure_nn_client_with_random_data,
                    args=[secure_nn, correctness_func, master_address, master_port])
        p.start()
        processes.append(p)
        for p in processes:
            p.join()

    if party == Config.server_rank:
        run_secure_nn_server_with_random_data(secure_nn, correctness_func, master_address, master_port)
    if party == Config.client_rank:
        run_secure_nn_client_with_random_data(secure_nn, correctness_func, master_address, master_port)

    print(f"\nTest for {test_name}: End")

if __name__ == "__main__":
    input_sid, master_addr, master_port, test_to_run = argparser_distributed()
    sys.stdout = Logger()

    print("====== New Tests ======")
    print("Test To run:", test_to_run)

    num_repeat = 5

    for _ in range(num_repeat):
        if test_to_run in ["small", "all"]:
            marshal_secure_nn_parties(input_sid, master_addr, master_port, generate_small_nn(), correctness_small_nn)
        if test_to_run in ["relu", "all"]:
            marshal_secure_nn_parties(input_sid, master_addr, master_port, generate_relu_only_nn(), correctness_relu_only_nn)
        if test_to_run in ["maxpool", "all"]:
            marshal_secure_nn_parties(input_sid, master_addr, master_port, generate_maxpool2x2(), correctness_maxpool2x2)
        if test_to_run in ["conv2d", "all"]:
            marshal_secure_nn_parties(input_sid, master_addr, master_port, generate_conv2d(), correctness_conv2d)
        if test_to_run in ["fc", "all"]:
            marshal_secure_nn_parties(input_sid, master_addr, master_port, generate_fc(), correctness_fc)
        if test_to_run in ["minionn_cifar", "all"]:
            marshal_secure_nn_parties(input_sid, master_addr, master_port, generate_minionn_cifar10(), correctness_minionn_cifar10)
        if test_to_run in ["minionn_mnist", "all"]:
            marshal_secure_nn_parties(input_sid, master_addr, master_port, generate_minionn_mnist(), correctness_minionn_mnist)
        if test_to_run in ["vgg", "all"]:
            marshal_secure_nn_parties(input_sid, master_addr, master_port, generate_vgg(), correctness_vgg)
