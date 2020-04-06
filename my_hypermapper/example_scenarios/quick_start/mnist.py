#!/usr/bin/python
import pandas as pd
import math
import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
import time

import numpy as np
import torch.utils.data as Data

#tds
import torchvision

#git
#from __future__ import print_function
#import argparse
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# from torch.optim.lr_scheduler import StepLR

import json

from subprocess import Popen, PIPE
# sys.path.append('path/to/hypermapper/scripts')
sys.path.append('scripts')

import hypermapper

# Load the MNIST dataset
# I had issues with batch size because torch.utils only accepts batch size in type int
# and not numpy.int64 (standard type for integers in numpy.array()).
# This I think is poor design but easy to work around
def get_mnist(b_size = 64):
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('data/', train=True,
                                  download=True,
                                  transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
                                  batch_size = b_size,
                                    shuffle=True)

    test_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('data/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
                                  batch_size = 1000, shuffle=True)
    # print('loaded the mnist data')
    return train_loader, test_loader


# Translates HyperMapper json to pyTorch module
class json2pheno(nn.Module):
    def __init__(self, json, nin, nout):
        super(json2pheno, self).__init__()
        # build layers from genome encoding

        n_in = nin
        fw_map = {}

        """ What does json contain that we want here:
        n_nodes
        n_layers (All layers accept the output layer)
        skip (length of skip connection. may be 0)
        """
        n_nodes = json['n_nodes']
        n_layers = json['n_layers']
        skip = json['skip']
        if 'activation' in json:
            self.activation = json['activation']
        else:
            self.activation = 0

        # check so activation function is right
        if self.activation == 0:
            print('activation: tanh')
        elif self.activation == 1:
            print('activation: relu')
        else:
            print('error in act func')

        for i in range(n_layers):
            key = str(i)
            setattr(self, key, nn.Linear(n_in, n_nodes))

            # We are on the last hidden layer, so we will not have any skipps here
            if i == n_layers - 1:
                fw_map[key] = ['out']
                n_in = n_nodes
                break

            fw_map[key] = [str(i + 1)]

            # Add skips to the fw_map. If they are to long, sent them to output layer
            if skip:
                if i + skip + 1 < n_layers:
                    fw_map[key].append(str(i + skip + 1))
                else:
                    fw_map[key].append('out')

            # Again, this is same for all but first layer
            n_in = n_nodes

        setattr(self, 'out', nn.Linear(n_in, nout))

        # fw_scheme is a dict containing to which layers each layer is sending its output
        # This will fail if we have non-forward connections
        self.fw_scheme = fw_map
        print(self.fw_scheme)

    def forward(self, x):
        k = 0
        X = dict()
        X['0'] = [x.view(x.shape[0], -1)]
        while hasattr(self, str(k)):
            # pass trough all layers except the output layer
            key = str(k)

            # we might want to concat instead of sum, then we need to modify input_size in __init__
            temp_x = sum(X[key])
            # temp_out = torch.tanh(getattr(self, key)(temp_x))
            if self.activation == 0:
                temp_out = torch.tanh(getattr(self, key)(temp_x))
            elif self.activation == 1:
                temp_out = torch.relu(getattr(self, key)(temp_x))
            else:
                temp_out = getattr(self, key)(temp_x)

            # this seem to work when doing the list thing with x
            for target in self.fw_scheme[key]:
                if target in X:
                    X[target].append(temp_out)
                else:
                    X[target] = [temp_out]

            k += 1

        # if k = 0 we have no active layers and a perceptron model
        if k:
            temp_x = sum(X['out'])
        else:
            temp_x = x.view(x.shape[0], -1)

        # Softmax as we are dealing with multiclass clasification problems
        out = getattr(self, 'out')(temp_x)
        return F.log_softmax(out, dim=1)


# Trains network and returns validation performance
def trainer(network, train_data, test_data, device, optimizer=0):
    # Always uses cross entropy as loss function
    criterion = nn.CrossEntropyLoss()

    # Optimization algorithm given by the json
    if optimizer == 0:
        print('optimizer: sgd')
        # SGD requires lr to be set
        optimizer = optim.SGD(network.parameters(), lr=0.1)
    elif optimizer == 1:
        print('optimizer: adam')
        optimizer = optim.Adam(network.parameters())
    elif optimizer == 2:
        print('optimizer: rmsprop')
        optimizer = optim.RMSprop(network.parameters())

    # Train for a given number of epochs (1)
    t0 = time.perf_counter()
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_data, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # running_loss += loss.item()
            # if i % 200 == 199:    # print every 200 mini-batches
            #   print('[%d, %5d] loss: %.3f' %
            #        (epoch + 1, i + 1, running_loss / 200))
            # running_loss = 0.0

    print('Finished Training it took ', (time.perf_counter() - t0) / 60, ' minutes to train')

    # Validates performance on unseen data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_data:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #   100 * correct / total))

    # Accuracy is our performance measure, err = 1 / acc gives us a minimization problem. Which is preferred.
    accuracy = correct / total
    return 1 / accuracy


# Objective function for HyperMapper to optimize
def MNIST_function(X):
    """
    Compute the error rate on MNIST after training for a given time.
    :param X: dictionary containing the hyperparameters describing a network.
    :return: the validation performance of the network described by X
    """

    # Sets batch size if given in json. Otherwise defaults to 64
    # Gets issues with type when taken from array. Then it must be converted to int
    if 'batch_size' in X:
        batch_size = X['batch_size']
        if type(batch_size) != int:
            # print('not a pos int!! type: ', type(batch_size))
            batch_size = int(batch_size)
            # print('Now it should be int. type: ', type(batch_size))

        print(batch_size)
        train_loader, test_loader = get_mnist(batch_size)
    else:
        print('batch size not given, defaults to 64')
        train_loader, test_loader = get_mnist()

    # The input/output size of the network.
    # for fully connected nets nin is nbr of pixels
    # for CNN nin would be number of chanels
    nin = 28 ** 2
    nout = 10
    my_net = json2pheno(X, nin, nout)

    ### GPU ###
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print('device: ', device.type)
    my_net.to(device)

    # Specifies optimizer if given in json scenario.
    if 'optimizer' in X:
        optimizer = X['optimizer']
        loss = trainer(my_net, train_loader, test_loader, device, optimizer)
    else:
        print('optimizer not given')
        loss = trainer(my_net, train_loader, test_loader, device)

    print('error: ', loss)
    print('\n')
    return loss


def main():
    # Change output path in scenario if you are running something serious
    # Right now we run 10 RS and saves output in stupid.csv
    parameters_file = "example_scenarios/quick_start/mnist_scenario.json"

    t_start = time.perf_counter()
    # HyperMapper runs the optimization procedure with MNIST_function as objective and parameters_file as Search Space
    hypermapper.optimize(parameters_file, MNIST_function)
    print('this entire procedure took ', (time.perf_counter() - t_start) / 60, 'minutes')
    print('End of MNIST')


if __name__ == "__main__":
    main()