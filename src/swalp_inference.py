from __future__ import print_function

import os
import sys
import time
from collections import namedtuple
import math

from model_reading import get_net_config_name
from modulus_net import data_shift, ModulusNet, ModulusNet_vgg16, ModulusNet_MiniONN

from torch import nn
from torch.autograd import Function
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

from logger_utils import Logger
from torch_utils import argparser_distributed, nmod, adjust_learning_rate, load_state_dict, warming_up_cuda

dtype = torch.float
device = torch.device("cuda:0")

#modulus = 8273921

minibatch = 128
store_configs = {}

def main(net_state_name, net, testset):
    warming_up_cuda()
    # print("CUDA is available:", torch.cuda.is_available())

    testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch, shuffle=False, num_workers=4)

    # seed = 100
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)

    net.to(device)

    sys.stdout = Logger()

    NumShowInter = 100
    NumEpoch = 200
    IterCounter = -1
    training_start = time.time()

    load_state_dict(net_state_name, net)
    sys.stdout = Logger()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)

    correct = 0
    total = 0

    loss_sum = 0.0
    cnt = 0
    inference_start = time.time()
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            loss = criterion(outputs, labels)
            loss_sum += loss.data.cpu().item() * images.size(0)
            correct += (predicted == labels).sum().item()
            cnt += int(images.size()[0])
        print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))
        print("loss=", loss_sum/float(cnt))

    elapsed_time = time.time() - inference_start
    print("Elapsed time for Prediction", elapsed_time)


if __name__ == "__main__":
    _, _, _, test_to_run = argparser_distributed()
    print("====== New Tests ======")
    print("Test To run:", test_to_run)

    net_state_name, config_name = get_net_config_name(test_to_run)
    print(f"net_state going to load: {net_state_name}")
    print(f"store_configs going to load: {config_name}")
    store_configs = np.load(config_name, allow_pickle="TRUE").item()

    def input_shift(data):
        first_layer_name = "conv1"
        return data_shift(data, store_configs[first_layer_name + "ForwardX"])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            input_shift
        ])

    if test_to_run in ["vgg16_cifar10", "minionn_maxpool", "minionn_cifar10"]:
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        if test_to_run == "vgg16_cifar10":
            main(net_state_name, ModulusNet_vgg16(store_configs, 10), testset)
        if test_to_run == "minionn_maxpool":
            main(net_state_name, ModulusNet_MiniONN(store_configs), testset)
        if test_to_run == "minionn_cifar10":
            main(test_to_run, NetQ_MiniONN(), testset)
    if test_to_run in ["vgg_cifar100", "res34_cifar100", "vgg16_cifar100", "vgg19_cifar100"]:
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        if test_to_run == "vgg_cifar100":
            main(test_to_run, NetQ_vgg(100), testset)
        if test_to_run == "res34_cifar100":
            main(test_to_run, NetQ_res34(100), testset)
        if test_to_run == "vgg16_cifar100":
            main(net_state_name, ModulusNet_vgg16(store_configs, 100), testset)
        if test_to_run == "vgg19_cifar100":
            main(test_to_run, NetQ_vgg19(100), testset)
