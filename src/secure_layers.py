from typing import Callable

import torch
import torch.nn.functional as F

from avgpool2x2 import Avgpool2x2Server, Avgpool2x2Client, Avgpool2x2Common
from comm import NamedBase, end_communicate, torch_sync, init_communicate, CommFheBuilder, BlobTorch, CommBase
from config import Config
from conv2d_ntt import calc_output_hw, Conv2dSecureCommon, Conv2dSecureServer, Conv2dSecureClient
from fhe import FheBuilder
from fully_connected import FcSecureCommon, FcSecureServer, FcSecureClient
from maxpool2x2_dgk import Maxpool2x2DgkServer, Maxpool2x2DgkClient, Maxpool2x2DgkCommon
from relu_dgk import ReluDgkCommon, ReluDgkServer, ReluDgkClient
from secret_share import ReconToClientServer, ReconToClientClient
from swap_share import SwapToClientOfflineCommon, SwapToClientOfflineServer, SwapToClientOfflineClient
from timer_utils import NamedTimerInstance
from torch_utils import get_prod, get_torch_size, marshal_funcs, gen_unirand_int_grain, pmod, compare_expected_actual, \
    generate_random_mask, nmod
from truncation import TruncCommon, TruncServer, TruncClient


class SecureLayerContext(NameError):
    class_name = "SecureLayerContext"
    fhe_builder_16: FheBuilder
    fhe_builder_23: FheBuilder
    rank: int

    def __init__(self, work_bit, data_bit, q_16, q_23, fhe_builder_16, fhe_builder_23, name):
        self.work_bit = work_bit
        self.data_bit = data_bit
        self.q_16 = q_16
        self.q_23 = q_23
        self.fhe_builder_16 = fhe_builder_16
        self.fhe_builder_23 = fhe_builder_23
        self.name = name

        self.data_range = 2 ** self.data_bit

    def set_rank(self, rank):
        self.rank = rank


class ContextRankBase(object):
    context: SecureLayerContext
    rank: int

    def __init__(self):
        pass

    def load_context(self, context: SecureLayerContext):
        self.context = context

    def is_server(self):
        return self.context.rank == Config.server_rank

    def is_client(self):
        return self.context.rank == Config.client_rank


class SecureLayerBase(NamedBase, ContextRankBase):
    class_name = "SecureLayerBase"
    prev_layer: "SecureLayerBase"
    next_layer: "SecureLayerBase"
    input_shape: torch.Size
    output_shape: torch.Size
    input_device: torch.device
    input_dtype: torch.dtype
    next_input_device: torch.device
    next_input_dtype: torch.dtype
    output_share: torch.Tensor
    reconstructed_output: torch.Tensor
    is_offline_known_s = False
    is_offline_known_c = False
    has_weight = False
    # hook: Callable[[torch.Tensor], None]

    def __init__(self, name):
        super().__init__(name)

    def register_next_layer(self, layer: "SecureLayerBase"):
        self.next_layer = layer
        self.next_input_device = layer.input_device
        self.next_input_dtype = layer.input_dtype

    def register_prev_layer(self, layer: "SecureLayerBase"):
        self.prev_layer = layer
        self.input_shape = layer.get_output_shape()

    # def register_hook(self, hook):
    #     self.hook = hook

    def offline(self):
        raise NotImplementedError()

    def online(self):
        raise NotImplementedError()

    def reconstructed_to_server(self, comm_base: CommBase, modulus):
        blob_output_share = BlobTorch(self.get_output_shape(), torch.float, comm_base, self.name + "_output_share")

        if self.is_server():
            blob_output_share.prepare_recv()
            torch_sync()
            other_output_share = blob_output_share.get_recv()
            # print(self.name + "_output_share" + "_server: have", self.get_output_share())
            # print(self.name + "_output_share" + "_server: received", other_output_share)
            self.reconstructed_output = nmod(self.get_output_share() + other_output_share, modulus)
            # print(self.name + "_output_share" + "_server: recon", self.reconstructed_output)

        if self.is_client():
            torch_sync()
            blob_output_share.send(self.get_output_share())
            # print(self.name + "_output_share" + "_client: sent", self.get_output_share())

    def get_reconstructed_output(self):
        return self.reconstructed_output

    def get_output_shape(self):
        return self.output_shape

    def get_input_share(self):
        return self.prev_layer.get_output_share()

    def get_output_share(self):
        return self.output_share


class InputSecureLayer(SecureLayerBase):
    class_name = "InputSecureLayer"
    input_device = torch.device("cpu")
    input_dtype = torch.float
    input_img: torch.Tensor
    swap_prot: SwapToClientOfflineCommon
    dummy_input_s: torch.Tensor
    is_offline_known_c = True

    def __init__(self, shape, name):
        super().__init__(name)
        self.input_shape = get_torch_size(shape)

    def offline(self):
        device = self.next_input_device
        dtype = self.next_input_dtype
        swap_prot_name = self.sub_name("swap_prot")
        modulus = self.context.q_23
        if self.is_server():
            self.swap_prot = SwapToClientOfflineServer(get_prod(self.input_shape), modulus, swap_prot_name)
            self.dummy_input_s = torch.zeros(self.input_shape).to(device).type(dtype)
            self.swap_prot.offline()
        elif self.is_client():
            self.swap_prot = SwapToClientOfflineClient(get_prod(self.input_shape), modulus, swap_prot_name)
            self.output_share = generate_random_mask(modulus, self.input_shape)
            self.swap_prot.offline(self.output_share.reshape(-1))
            self.output_share = self.output_share.to(device).type(dtype)

    def online(self):
        device = self.next_input_device
        dtype = self.next_input_dtype
        if self.is_client():
            if self.input_img is None:
                raise Exception("Client should feed input")
            self.swap_prot.online(self.input_img.reshape(-1))
            self.output_share = self.swap_prot.output_c
        elif self.is_server():
            self.swap_prot.online(self.dummy_input_s.reshape(-1))
            self.output_share = self.swap_prot.output_s
        self.output_share = self.output_share.to(device).type(dtype).reshape(self.input_shape)

    def feed_input(self, input_img: torch.Tensor):
        assert(input_img.shape == self.input_shape)
        assert(self.is_client())
        self.input_img = input_img

    def get_output_shape(self):
        return self.input_shape

    def get_output_share(self):
        return self.output_share


class OutputSecureLayer(SecureLayerBase):
    class_name = "OutputSecureLayer"
    input_device = torch.device("cpu")
    input_dtype = torch.float
    output: torch.Tensor
    prot = None

    def __init__(self, name):
        super().__init__(name)

    def offline(self):
        num_elem = get_prod(self.input_shape)
        modulus = self.context.q_23
        name = self.class_name + self.name
        if self.is_server():
            self.prot = ReconToClientServer(num_elem, modulus, name)
        elif self.is_client():
            self.prot = ReconToClientClient(num_elem, modulus, name)

        self.prot.offline()

    def online(self):
        assert(self.get_input_share().shape == self.input_shape)
        self.prot.online(self.get_input_share().reshape(-1))
        if self.is_client():
            self.output = self.prot.output.reshape(self.input_shape)

    def get_output_shape(self):
        return self.input_shape

    def get_output(self):
        return self.output


class FlattenSecureLayer(SecureLayerBase):
    class_name = "FlattenSecureLayer"
    input_device = None
    input_dtype = None

    def __init__(self, name):
        super().__init__(name)

    def register_prev_layer(self, layer: SecureLayerBase):
        SecureLayerBase.register_prev_layer(self, layer)
        self.output_shape = get_torch_size(get_prod(self.input_shape))

    def register_next_layer(self, layer: SecureLayerBase):
        SecureLayerBase.register_next_layer(self, layer)
        self.input_device = self.next_input_device
        self.input_dtype = self.next_input_dtype
        self.prev_layer.register_next_layer(self)

    def offline(self):
        return

    def online(self):
        assert(self.get_input_share().shape == self.input_shape)
        self.output_share = self.get_input_share().reshape(self.output_shape)


class SwapToClientOfflineLayer(SecureLayerBase):
    class_name = "SwapToClientOfflineLayer"
    input_device = torch.device("cuda")
    input_dtype = torch.float
    swap_prot: SwapToClientOfflineCommon
    swapped_input_s: torch.Tensor
    swapped_input_c: torch.Tensor
    is_need_swap = True

    def __init__(self, name):
        super().__init__(name)

    def offline(self):
        modulus = self.context.q_23
        swap_prot_name = self.sub_name("swap_prot")

        if self.is_need_swap:
            if self.is_server():
                self.swap_prot = SwapToClientOfflineServer(get_prod(self.input_shape), modulus, swap_prot_name)
                self.swap_prot.offline()
            elif self.is_client():
                self.swap_prot = SwapToClientOfflineClient(get_prod(self.input_shape), modulus, swap_prot_name)
                self.swapped_input_c = generate_random_mask(modulus, self.input_shape)
                self.swap_prot.offline(self.swapped_input_c.reshape(-1))
                self.swapped_input_c = self.swapped_input_c.to(Config.device).reshape(self.input_shape)
        if not self.is_need_swap and self.is_client():
            self.swapped_input_c = self.get_input_share().to(Config.device)

    def online(self):
        if self.is_need_swap:
            if self.is_server():
                self.swap_prot.online(self.get_input_share().reshape(-1))
                self.swapped_input_s = self.swap_prot.output_s.reshape(self.input_shape)
            elif self.is_client():
                self.swap_prot.online(self.get_input_share().reshape(-1))
                self.swapped_input_c = self.swapped_input_c.to(Config.device).reshape(self.input_shape)
        if not self.is_need_swap and self.is_server():
            self.swapped_input_s = self.get_input_share()


class Conv2dSecureLayer(SwapToClientOfflineLayer):
    class_name = "Conv2dSecureLayer"
    input_device = torch.device("cuda")
    input_dtype = torch.float
    is_offline_known_c = True
    has_weight = True
    img_hw: int
    output_hw: int
    compute_prot: Conv2dSecureCommon
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, num_input_channel, num_output_channel, filter_hw, name, padding=1, bias=None):
        super().__init__(name)
        self.num_input_channel = num_input_channel
        self.num_output_channel = num_output_channel
        self.filter_hw = filter_hw
        self.padding = padding
        self.bias = bias

        self.weight_shape = torch.Size([num_output_channel, num_input_channel, filter_hw, filter_hw])
        self.bias_shape = torch.Size([num_output_channel])

    def register_prev_layer(self, layer: "SecureLayerBase"):
        SecureLayerBase.register_prev_layer(self, layer)
        self.input_shape = layer.get_output_shape()
        assert(len(self.input_shape) == 3)
        assert(self.input_shape[-3] == self.num_input_channel)
        assert(self.input_shape[-1] == self.input_shape[-2])
        self.img_hw = self.input_shape[-2]
        self.output_hw = calc_output_hw(self.img_hw, self.filter_hw, self.padding)
        self.output_shape = torch.Size([self.num_output_channel, self.output_hw, self.output_hw])
        if layer.is_offline_known_c:
            self.is_need_swap = False

    def load_weight(self, weight, bias=None):
        self.weight = weight
        self.bias = bias

    def offline(self):
        SwapToClientOfflineLayer.offline(self)

        modulus = self.context.q_23
        compute_prot_name = self.sub_name("compute_prot")
        fhe_builder = self.context.fhe_builder_23
        data_range = self.context.data_range

        if self.is_server():
            self.compute_prot = Conv2dSecureServer(modulus, fhe_builder, data_range, self.img_hw, self.filter_hw,
                                                   self.num_input_channel, self.num_output_channel,
                                                   compute_prot_name, padding=self.padding)
            self.compute_prot.offline(self.weight, bias=self.bias)
        elif self.is_client():
            self.compute_prot = Conv2dSecureClient(modulus, fhe_builder, data_range, self.img_hw, self.filter_hw,
                                                   self.num_input_channel, self.num_output_channel,
                                                   compute_prot_name, padding=self.padding)
            self.compute_prot.offline(self.swapped_input_c)
            self.output_share = self.compute_prot.output_c

    def online(self):
        SwapToClientOfflineLayer.online(self)
        if self.is_server():
            self.compute_prot.online(self.swapped_input_s)
            self.output_share = self.compute_prot.output_s
        elif self.is_client():
            self.compute_prot.online()


class FcSecureLayer(SwapToClientOfflineLayer):
    class_name = "FcSecureLayer"
    input_device = torch.device("cuda")
    input_dtype = torch.float
    is_offline_known_c = True
    has_weight = True
    img_hw: int
    output_hw: int
    compute_prot: FcSecureCommon
    weight: torch.Tensor
    bias: torch.Tensor
    num_input_unit: int
    num_output_unit: int

    def __init__(self, num_output_unit, name):
        super().__init__(name)
        self.num_output_unit = num_output_unit

    def register_prev_layer(self, layer: "SecureLayerBase"):
        SecureLayerBase.register_prev_layer(self, layer)
        self.input_shape = layer.get_output_shape()
        assert(len(self.input_shape) == 1)
        self.num_input_unit = self.input_shape[-1]
        self.weight_shape = torch.Size([self.num_output_unit, self.num_input_unit])
        self.output_shape = torch.Size([self.num_output_unit])
        self.bias_shape = torch.Size([self.num_output_unit])

        if layer.is_offline_known_c:
            self.is_need_swap = False

    def load_weight(self, weight, bias=None):
        self.weight = weight
        self.bias = bias

    def offline(self):
        SwapToClientOfflineLayer.offline(self)

        modulus = self.context.q_23
        compute_prot_name = self.sub_name("compute_prot")
        fhe_builder = self.context.fhe_builder_23
        data_range = self.context.data_range

        if self.is_server():
            self.compute_prot = FcSecureServer(modulus, data_range, self.num_input_unit, self.num_output_unit,
                                               fhe_builder, compute_prot_name)
            self.compute_prot.offline(self.weight, bias=self.bias)
        elif self.is_client():
            self.compute_prot = FcSecureClient(modulus, data_range, self.num_input_unit, self.num_output_unit,
                                               fhe_builder, compute_prot_name)
            self.compute_prot.offline(self.swapped_input_c)
            self.output_share = self.compute_prot.output_c

    def online(self):
        SwapToClientOfflineLayer.online(self)
        if self.is_server():
            self.compute_prot.online(self.swapped_input_s)
            self.output_share = self.compute_prot.output_s
        elif self.is_client():
            self.compute_prot.online()


class TruncSecureLayer(SecureLayerBase):
    class_name = "TruncSecureLayer"
    input_device = torch.device("cuda")
    input_dtype = torch.float
    prot: TruncCommon
    div_to_pow: int

    def __init__(self, name):
        super().__init__(name)

    def register_prev_layer(self, layer: SecureLayerBase):
        SecureLayerBase.register_prev_layer(self, layer)
        self.input_shape = layer.get_output_shape()
        self.output_shape = self.input_shape

    def set_div_to_pow(self, div_to_pow):
        self.div_to_pow = div_to_pow

    def offline(self):
        num_elem = get_prod(self.input_shape)
        name = self.sub_name("trunc_prot")

        if self.is_server():
            self.prot = TruncServer(num_elem, self.context.q_23, self.div_to_pow, self.context.fhe_builder_23, name)
        elif self.is_client():
            self.prot = TruncClient(num_elem, self.context.q_23, self.div_to_pow, self.context.fhe_builder_23, name)

        self.prot.offline()

    def online(self):
        self.prot.online(self.get_input_share().reshape(-1))

        if self.is_server():
            self.output_share = self.prot.out_s
        elif self.is_client():
            self.output_share = self.prot.out_c

        device = self.next_input_device
        dtype = self.next_input_dtype
        self.output_share = self.output_share.to(device).type(dtype).reshape(self.output_shape)


class ReluSecureLayer(SecureLayerBase):
    class_name = "ReluSecureLayer"
    input_device = torch.device("cuda")
    input_dtype = torch.float
    prot: ReluDgkCommon

    def __init__(self, name):
        super().__init__(name)

    def register_prev_layer(self, layer: SecureLayerBase):
        SecureLayerBase.register_prev_layer(self, layer)
        self.input_shape = layer.get_output_shape()
        self.output_shape = self.input_shape

    def offline(self):
        num_elem = get_prod(self.input_shape)
        name = self.sub_name("relu_dgk_prot")

        if self.is_server():
            self.prot = ReluDgkServer(num_elem, self.context.q_23, self.context.q_16,
                                      self.context.work_bit, self.context.data_bit,
                                      self.context.fhe_builder_16, self.context.fhe_builder_23, name)
        elif self.is_client():
            self.prot = ReluDgkClient(num_elem, self.context.q_23, self.context.q_16,
                                      self.context.work_bit, self.context.data_bit,
                                      self.context.fhe_builder_16, self.context.fhe_builder_23, name)

        self.prot.offline()

    def online(self):
        self.prot.online(self.get_input_share().reshape(-1))

        if self.is_server():
            self.output_share = self.prot.max_s
        elif self.is_client():
            self.output_share = self.prot.max_c

        device = self.next_input_device
        dtype = self.next_input_dtype
        self.output_share =  self.output_share.to(device).type(dtype).reshape(self.output_shape)


class Maxpool2x2SecureLayer(SecureLayerBase):
    class_name = "Maxpool2x2SecureLayer"
    input_device = torch.device(Config.device)
    input_dtype = torch.float
    prot: Maxpool2x2DgkCommon
    input_hw: int
    output_hw: int
    num_channel: int

    def __init__(self, name):
        super().__init__(name)

    def register_prev_layer(self, layer: SecureLayerBase):
        SecureLayerBase.register_prev_layer(self, layer)
        self.input_shape = layer.get_output_shape()
        assert(len(self.input_shape) == 3)
        assert(self.input_shape[-1] == self.input_shape[-2])
        self.input_hw = self.input_shape[-1]
        assert(self.input_hw % 2 == 0)
        self.output_hw = self.input_hw // 2
        self.num_channel = self.input_shape[-3]
        self.output_shape = torch.Size([self.num_channel, self.output_hw, self.output_hw])

    def offline(self):
        num_elem = get_prod(self.input_shape)
        name = self.sub_name("maxpool2x2_dgk_prot")

        if self.is_server():
            self.prot = Maxpool2x2DgkServer(num_elem, self.context.q_23, self.context.q_16,
                                            self.context.work_bit, self.context.data_bit,
                                            self.input_hw,
                                            self.context.fhe_builder_16, self.context.fhe_builder_23, name)
        elif self.is_client():
            self.prot = Maxpool2x2DgkClient(num_elem, self.context.q_23, self.context.q_16,
                                            self.context.work_bit, self.context.data_bit,
                                            self.input_hw,
                                            self.context.fhe_builder_16, self.context.fhe_builder_23, name)

        self.prot.offline()

    def online(self):
        self.prot.online(self.get_input_share().reshape(-1))

        if self.is_server():
            self.output_share = self.prot.max_s
        elif self.is_client():
            self.output_share = self.prot.max_c

        device = self.next_input_device
        dtype = self.next_input_dtype
        self.output_share =  self.output_share.to(device).type(dtype).reshape(self.output_shape)


class Avgpool2x2SecureLayer(SecureLayerBase):
    class_name = "Avgpool2x2SecureLayer"
    input_device = torch.device("cuda")
    input_dtype = torch.float
    prot: Avgpool2x2Common
    input_hw: int
    output_hw: int
    num_channel: int

    def __init__(self, name):
        super().__init__(name)

    def register_prev_layer(self, layer: SecureLayerBase):
        SecureLayerBase.register_prev_layer(self, layer)
        self.input_shape = layer.get_output_shape()
        assert(len(self.input_shape) == 3)
        assert(self.input_shape[-1] == self.input_shape[-2])
        self.input_hw = self.input_shape[-1]
        assert(self.input_hw % 2 == 0)
        self.output_hw = self.input_hw // 2
        self.num_channel = self.input_shape[-3]
        self.output_shape = torch.Size([self.num_channel, self.output_hw, self.output_hw])

    def offline(self):
        num_elem = get_prod(self.input_shape)
        name = self.sub_name("avgpool2x2_prot")

        if self.is_server():
            self.prot = Avgpool2x2Server(num_elem, self.context.q_23, self.context.q_16,
                                         self.context.work_bit, self.context.data_bit,
                                         self.input_hw,
                                         self.context.fhe_builder_16, self.context.fhe_builder_23, name)
        elif self.is_client():
            self.prot = Avgpool2x2Client(num_elem, self.context.q_23, self.context.q_16,
                                         self.context.work_bit, self.context.data_bit,
                                         self.input_hw,
                                         self.context.fhe_builder_16, self.context.fhe_builder_23, name)

        self.prot.offline()

    def online(self):
        self.prot.online(self.get_input_share().reshape(-1))

        if self.is_server():
            self.output_share = self.prot.max_s
        elif self.is_client():
            self.output_share = self.prot.max_c

        device = self.next_input_device
        dtype = self.next_input_dtype
        self.output_share =  self.output_share.to(device).type(dtype).reshape(self.output_shape)


class SecureNeuralNetwork(NamedBase, ContextRankBase):
    class_name = "SecureNeuralNetwork"
    context: SecureLayerContext
    input_layer: InputSecureLayer
    output_layer: OutputSecureLayer
    layers: list

    def __init__(self, name):
        super().__init__(name)

    def load_layers(self, layers):
        self.layers = layers

        if not isinstance(self.layers[0], InputSecureLayer):
            raise ValueError("The first layer has to be input layer")
        if not isinstance(self.layers[-1], OutputSecureLayer):
            raise ValueError("The last layer has to be output layer")

        self.input_layer = layers[0]
        self.output_layer = layers[-1]

        for i in range(len(self.layers) - 1):
            prev_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            prev_layer.register_next_layer(next_layer)
            next_layer.register_prev_layer(prev_layer)

    def load_context(self, context: SecureLayerContext):
        ContextRankBase.load_context(self, context)
        self.context = context

        for layer in self.layers:
            layer.load_context(context)

    def feed_input(self, img):
        if self.is_client():
            self.input_layer.feed_input(img)
        else:
            raise Exception("Only the client, not the server, can input.")

    def offline(self):
        party = "Server" if self.is_server() else "Client"
        for layer in self.layers:
            with NamedTimerInstance(f"{party} Offline of {layer.name}"):
                layer.offline()
                torch_sync()

    def online(self):
        party = "Server" if self.is_server() else "Client"
        for layer in self.layers:
            with NamedTimerInstance(f"{party} Online of {layer.name}"):
                layer.online()
                # torch_sync()

    def get_output(self):
        return self.output_layer.get_output()

    def get_argmax_output(self):
        _, predicted = torch.max(self.get_output(), 1)
        return predicted


def test_secure_nn():
    test_name = "test_secure_nn"
    print(f"\nTest for {test_name}: Start")
    data_bit = 5
    work_bit = 17
    data_range = 2 ** data_bit
    q_16 = 12289
    # q_23 = 786433
    q_23 = 7340033
    # q_23 = 8273921
    input_img_hw = 16
    input_channel = 3
    pow_to_div = 2

    fhe_builder_16 = FheBuilder(q_16, 2048)
    fhe_builder_23 = FheBuilder(q_23, 8192)

    input_shape = [input_channel, input_img_hw, input_img_hw]

    context = SecureLayerContext(work_bit, data_bit, q_16, q_23, fhe_builder_16, fhe_builder_23, test_name+"_context")

    input_layer = InputSecureLayer(input_shape, "input_layer")
    conv1 = Conv2dSecureLayer(3, 5, 3, "conv1", padding=1)
    relu1 = ReluSecureLayer("relu1")
    trunc1 = TruncSecureLayer("trunc1")
    pool1 = Maxpool2x2SecureLayer("pool1")
    conv2 = Conv2dSecureLayer(5, 10, 3, "conv2", padding=1)
    flatten = FlattenSecureLayer("flatten")
    fc1 = FcSecureLayer(32, "fc1")
    output_layer = OutputSecureLayer("output_layer")

    secure_nn = SecureNeuralNetwork("secure_nn")
    secure_nn.load_layers([input_layer, conv1, pool1, relu1, trunc1, conv2, flatten, fc1, output_layer])
    # secure_nn.load_layers([input_layer, relu1, trunc1, output_layer])
    secure_nn.load_context(context)

    def generate_random_data(shape):
        return gen_unirand_int_grain(-data_range//2 + 1, data_range//2, get_prod(shape)).reshape(shape)

    conv1_w = generate_random_data(conv1.weight_shape)
    conv2_w = generate_random_data(conv2.weight_shape)
    fc1_w = generate_random_data(fc1.weight_shape)

    def check_correctness(input_img, output):
        torch_pool1 = torch.nn.MaxPool2d(2)

        x = input_img.to(Config.device).double()
        x = x.reshape([1] + list(x.shape))
        x = pmod(F.conv2d(x, conv1_w.to(Config.device).double(), padding=1), q_23)
        x = pmod(F.relu(nmod(x, q_23)), q_23)
        x = pmod(torch_pool1(nmod(x, q_23)), q_23)
        x = pmod(x // (2 ** pow_to_div), q_23)
        x = pmod(F.conv2d(x, conv2_w.to(Config.device).double(), padding=1), q_23)
        x = x.view(-1)
        x = pmod(torch.mm(x.view(1, -1), fc1_w.to(Config.device).double().t()).view(-1), q_23)

        expected = x
        actual = pmod(output, q_23)
        if len(expected.shape) == 4 and expected.shape[0] == 1:
            expected = expected.reshape(expected.shape[1:])
        compare_expected_actual(expected, actual, name=test_name, get_relative=True)

    def test_server():
        rank = Config.server_rank
        init_communicate(rank)
        context.set_rank(rank)

        comm_fhe_16 = CommFheBuilder(rank, fhe_builder_16, "fhe_builder_16")
        comm_fhe_23 = CommFheBuilder(rank, fhe_builder_23, "fhe_builder_23")
        comm_fhe_16.recv_public_key()
        comm_fhe_23.recv_public_key()
        comm_fhe_16.wait_and_build_public_key()
        comm_fhe_23.wait_and_build_public_key()

        conv1.load_weight(conv1_w)
        conv2.load_weight(conv2_w)
        fc1.load_weight(fc1_w)
        trunc1.set_div_to_pow(pow_to_div)

        with NamedTimerInstance("Server Offline"):
            secure_nn.offline()
            torch_sync()

        with NamedTimerInstance("Server Online"):
            secure_nn.online()
            torch_sync()

        end_communicate()

    def test_client():
        rank = Config.client_rank
        init_communicate(rank)
        context.set_rank(rank)

        fhe_builder_16.generate_keys()
        fhe_builder_23.generate_keys()
        comm_fhe_16 = CommFheBuilder(rank, fhe_builder_16, "fhe_builder_16")
        comm_fhe_23 = CommFheBuilder(rank, fhe_builder_23, "fhe_builder_23")
        comm_fhe_16.send_public_key()
        comm_fhe_23.send_public_key()

        input_img = generate_random_data(input_shape)
        trunc1.set_div_to_pow(pow_to_div)
        secure_nn.feed_input(input_img)

        with NamedTimerInstance("Client Offline"):
            secure_nn.offline()
            torch_sync()

        with NamedTimerInstance("Client Online"):
            secure_nn.online()
            torch_sync()

        check_correctness(input_img, secure_nn.get_output())
        end_communicate()

    marshal_funcs([test_server, test_client])
    print(f"\nTest for {test_name}: End")

if __name__ == "__main__":
    test_secure_nn()
