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
# from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader

import json
from subprocess import Popen, PIPE

#import matplotlib.pyplot as plt



# sys.path.append('path/to/hypermapper/scripts')
sys.path.append('scripts')

import hypermapper

# Load the MNIST dataset
def get_data_loaders(train_batch_size, test_batch_size, size=(224,224)):

    mnist = MNIST('data/', download=False, train=True).train_data.float()

    data_transform = Compose([ Resize(size) ,ToTensor(), Normalize((mnist.mean()/255,), (mnist.std()/255,))])

    train_loader = DataLoader(MNIST('data/', download=True, transform=data_transform, train=True),
                              batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(MNIST('data/', download=False, transform=data_transform, train=False),
                            batch_size=test_batch_size, shuffle=False)

    # print('loaded the mnist data')
    return train_loader, test_loader


# counts trainable weights in a model
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Vill ha bilder av storlek 224x224
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
    # dilation=1, groups=1, bias=True, padding_mode='zeros')
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        planes is the number of filters we want
        inplanes can differ from planes as we can get inputs from multiple places
        """
        super(BasicBlock, self).__init__()

        norm_layer = nn.BatchNorm2d
        # groups=1,
        # base_width=64
        # dilation = 1
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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
    def __init__(self, block, filters, filter_upd, blocks, kernel_size=0, pool=0, reduce=0):
        super(json2ResNet, self).__init__()

        num_classes = 10
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = filters  # [0]  # 64

        # self.groups = 1    # This is for ResNeXt, but only used for bottlenecks..
        # self.base_width = 64
        # Filter before residual layers
        if kernel_size == 0:
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1,
                                   bias=False)
        else:
            # Halves dimension
            reduce *= 2
            self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
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

        self.fc = nn.Linear(filters * block.expansion * pixels, num_classes)

        # What is this??? weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        # previous_dilation = 1
        # dilate=False

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion  # block expansion = 1 for basic block

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
def trainer(network, train_data, test_data, epochs=1):
    # Always uses cross entropy as loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters())

    # Train for a given number of epochs (1)
    t0 = time.perf_counter()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

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
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training it took ', (time.perf_counter() - t0) / 60, ' minutes to train')

    # Validates performance on unseen data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data:
            images, labels = data
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
    train_loader, test_loader = get_data_loaders(batch_size_train, batch_size_test, size=size)

    # nbr_layers = X['n_layers']
    # filters = [X['n_filters0']]
    blocks = []
    # We only use the n_layers first parameters. use the active-stategy?
    for idx in range(4):
        key = 'n_blocks' + str(idx + 1)
        blocks.append(X[key])

    filters = X['n_filters']
    filter_upd = X['filter_upd']
    # blocks = X['n_blocks']
    kernel_size = X['conv0']
    pool = X['pool']
    reduce = X['reduce']
    """if X['conv0'] = 0:
        kernel_size = 3
    else:
        kernel_size = 3
    """
    my_net = json2ResNet(BasicBlock, filters, filter_upd, blocks, kernel_size=kernel_size, pool=pool, reduce=reduce)
    # print(my_net)
    # print('parmas: ', count_params(my_net))
    # print('we got a resnet by num lay: ', nbr_layers, 'filters: ', filters, 'blocks: ', blocks)
    loss = trainer(my_net, train_loader, test_loader, epochs=1)
    """
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    outputs = my_net(images)
    import numpy.random as rd
    loss = rd.random()
    """
    print('accuracy: ', 100 - loss)
    # print('\n')
    return loss

def main():
    # Change output path in scenario if you are running something serious
    # Right now we run 10 RS and saves output in stupid.csv
    parameters_file = "example_scenarios/quick_start/resnet_scenario.json"

    t_start = time.perf_counter()
    # HyperMapper runs the optimization procedure with ResNet_function as objective and parameters_file as Search Space
    hypermapper.optimize(parameters_file, ResNet_function)
    print('this entire procedure took ', (time.perf_counter() - t_start) / 60, 'minutes')
    print('End of ResNet')


if __name__ == "__main__":
    main()

