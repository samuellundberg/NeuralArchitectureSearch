#!/usr/bin/python
import math
import sys
import time

import numpy as np
# import pandas as pd

import torch
import torch.nn as nn
# import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
#from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop, RandomHorizontalFlip
from torch.utils.data import DataLoader

import json
from subprocess import Popen, PIPE

#import matplotlib.pyplot as plt



# sys.path.append('path/to/hypermapper/scripts')
sys.path.append('scripts')

import hypermapper


# Load the dataset
def get_data_loaders(train_batch_size, test_batch_size, size=(32, 32)):
    # Augments data like 4.2 in resnetpaper, is normalization right??
    transform_train = Compose([RandomHorizontalFlip(), Resize(size), RandomCrop(32, padding=4),
                               ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = Compose([Resize(size), ToTensor(),
                              Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_loader = DataLoader(CIFAR10('data/', download=False, transform=transform_train, train=True),
                              batch_size=train_batch_size, shuffle=True, num_workers=2)

    test_loader = DataLoader(CIFAR10('data/', download=False, transform=transform_test, train=False),
                             batch_size=test_batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes


# counts trainable weights in a model
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
    # dilation=1, groups=1, bias=True, padding_mode='zeros')
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        norm_layer = nn.BatchNorm2d
        #width = int(planes * (base_width / 64.)) * groups
        width = planes      # we don't want more filters, just a sparser connection.
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        """
        planes is the number of filters we want
        inplanes can differ from planes as we can get inputs from multiple places
        """
        super(BasicBlock, self).__init__()

        norm_layer = nn.BatchNorm2d
        # base_width=64
        # dilation = 1
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups=groups)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride  # what is the point of this

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# Translates HyperMapper json to pyTorch module
class json2ResNet(nn.Module):
    # ResNet(BasicBlock, [2, 2, 2, 2])
    def __init__(self, block, filters, filter_upd, groups, blocks, kernel_size, pool, reduce):
        super(json2ResNet, self).__init__()

        num_classes = 10
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = filters  # [0]  # 64

        self.groups = groups    # This is for ResNeXt, but only used for bottlenecks..
        # self.base_width = 64
        # Filter before residual layers
        if kernel_size == 0:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        else:
            # Halves dimension
            reduce *= 2
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = pool
        if pool:
            # 2x2 pooling halves the dimension
            reduce *= 2

        # ResNet for NAS
        reslays = 0
        for i, b in enumerate(blocks):  # len blocks = 4
            if b > 0:
                reslays += 1
                key = 'layer' + str(reslays)
                if i == 0:
                    lay = self._make_layer(block, filters, b)
                else:
                    reduce *= 2
                    filters = int(np.round(filters * filter_upd))
                    lay = self._make_layer(block, filters, b, stride=2)

                setattr(self, key, lay)

        self.reslays = reslays

        # End phase, Improve this
        pixels = 1
        if reduce:
            if reduce <= 16:
                reduce *= 2
                pixels = int((32 / reduce) ** 2)
                self.avgpool = nn.AvgPool2d(kernel_size=2)
            else:
                self.avgpool = nn.AvgPool2d(kernel_size=1)

        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.reslays > 0:    # when bottlencks have no blocks they never gets expanded
            self.fc = nn.Linear(filters * block.expansion * pixels, num_classes)
        else:
            self.fc = nn.Linear(filters * pixels, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization: https://arxiv.org/pdf/1502.01852.pdf
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # this is how bn weights usualy are initialized acording tp pytorch
                # bach_norm: https://arxiv.org/pdf/1502.03167.pdf
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # This is not part of the resnet paper, too new
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        """
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        """

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, groups=self.groups))
        self.inplanes = planes * block.expansion  # block expansion = 1 for basic block

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool:
            mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            x = mp(x)
        # print(x.shape)

        # NAS pass
        for idx in range(self.reslays):
            key = 'layer' + str(idx + 1)
            # print('fp: ', key)
            x = getattr(self, key)(x)
            # print(x.shape)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # We do not need softmax since it is done inside nn.CrossEntropyLoss()
        return x

    def forward(self, x):
        return self._forward_impl(x)


# Trains network and returns validation performance
def trainer(network, train_data, test_data, device, epochs=1):
    # Always uses cross entropy as loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(network.parameters())
    # run the first epoch with lr of 0.01 if deep network (100 but not 50ish)
    optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.9, dampening=0,
                          weight_decay=0.0001, nesterov=False)
    """
    How I want the learning rate:
    lr = 0.1 for epoch <80 
    lr = 0.01 for epoch 80-120
    lr = 0.001 for epoch >120
    lr = 0.01 for first epoch to warm up training if network is deep
    """
    scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

    t0 = time.perf_counter()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        scheduler.step()
        for i, data in enumerate(train_data, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 200 mini-batches
                # print('[%d, %5d] loss: %.3f' %
                #       (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training it took ', (time.perf_counter() - t0) / 60, ' minutes to train')

    # Validates performance on unseen data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #   100 * correct / total))

    return 100 * (1 - correct / total)    # 1 - accuracy for minimization

# Objective function for HyperMapper to optimize
def ResNet_function(X):
    """
    Compute the error rate on MNIST after training for a given time.
    :param X: dictionary containing the hyperparameters describing a network.
    :return: the validation performance of the network described by X
    """

    # Do the proper preprocessing they do in section 4.2 in resnetpaper
    batch_size_train = 128
    batch_size_test = 1000
    size = (32, 32)  # ResNet is made for 224, Mnist is 28, Cifar-10 is 32
    train_loader, test_loader, _ = get_data_loaders(batch_size_train, batch_size_test, size=size)

    blocks = []
    # We only use the n_layers first parameters. use the active-stategy?
    for idx in range(4):
        key = 'n_blocks' + str(idx + 1)
        blocks.append(X[key])

    filters = X['n_filters']
    filter_upd = X['filter_upd']

    group_size = X['group_size']
    if filters / group_size > 1:
        groups = int(filters / group_size)
    else:
        groups = 1

    kernel_size = X['conv0']
    pool = X['pool']
    reduce = X['reduce']

    block = BasicBlock if X['block'] == 0 else Bottleneck

    # print(X)
    my_net = json2ResNet(block, filters, filter_upd, groups, blocks, kernel_size, pool, reduce)
    ### GPU ###
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    # print('device: ', device.type)
    my_net.to(device)

    # print(my_net)
    # print('parmas: ', count_params(my_net))
    # print('we got a resnet by num lay: ', nbr_layers, 'filters: ', filters, 'blocks: ', blocks)
    #loss = trainer(my_net, train_loader, test_loader, device, epochs=5)

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    outputs = my_net(images)
    import numpy.random as rd
    loss = rd.random()

    #print('accuracy: ', 100 - loss)
    # print('\n')
    return loss


def main():
    # Change output path in scenario if you are running something serious
    # Right now we run 10 RS and saves output in stupid.csv
    parameters_file = "example_scenarios/quick_start/resnext_scenario.json"

    t_start = time.perf_counter()
    # HyperMapper runs the optimization procedure with ResNet_function as objective and parameters_file as Search Space
    hypermapper.optimize(parameters_file, ResNet_function)
    print('this entire procedure took ', (time.perf_counter() - t_start) / 60, 'minutes')
    # print('End of ResNet')


if __name__ == "__main__":
    main()

