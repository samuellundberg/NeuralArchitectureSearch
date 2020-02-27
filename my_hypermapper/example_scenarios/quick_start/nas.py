#!/usr/bin/python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

import math


import tkinter
import matplotlib
import matplotlib.pyplot as plt

import time
import sys

from subprocess import Popen, PIPE
# sys.path.append('path/to/hypermapper/scripts')
sys.path.append('scripts')

import hypermapper


# Stole most of this online, should be improved uppon
def trainer(net, epochs, noise, objective='x2'):
    torch.manual_seed(1)  # reproducible

    x = torch.unsqueeze(torch.linspace(-1, 1, 20), dim=1)  # x data (tensor), shape=(100, 1)

    if objective == 'x2':
        y = x.pow(2) + noise * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
    elif objective == 'sinx':
        y = torch.sin(3 * 3.14 * x) + noise * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

    # torch can only train on Variable, so convert them to Variable
    x, y = Variable(x), Variable(y)
    # does this help me??
    torch.autograd.set_detect_anomaly(True)

    print(net)  # net architecture

    optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    # train the network
    t0 = time.perf_counter()

    for t in range(epochs):
        prediction = net(x)  # input x and predict based on x
        loss = loss_func(prediction, y)  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    loss = loss.data.numpy()
    tr_time = time.perf_counter() - t0
    # print(tr_time, "seconds to train")

    # view data
    #plt.figure(figsize=(10, 4))
    #plt.scatter(x.data.numpy(), y.data.numpy(), color="orange")
    #plt.plot(x.data.numpy(), prediction.data.numpy(), 'g-', lw=3)

    #plt.title('Regression Analysis')
    #plt.text(1.0, 0, 'Loss = %.4f' % loss,
             #fontdict={'size': 24, 'color': 'red'})
    #plt.show()

    return loss, tr_time


# I think it would be good to use blocks. Maybe check the paper about Cells
class json2pheno(nn.Module):
    def __init__(self, json, nin, nout):
        super(json2pheno, self).__init__()
        # build layers from genome encoding

        n_in = nin
        fw_map = {}

        # stupid loop but I want to know how deep the net will be
        active_list = []
        for param in json.keys():
            if param[:-2] == 'active' and json[param] == 1:
                active_list.append(param[-1])

        # Add the active layers in the encoding to the model
        for new_i, old_i in enumerate(active_list):
            key = str(new_i)
            setattr(self, key, nn.Linear(n_in, json['n_nodes']))

            # We are on the last hidden layer, so we will not have any skipps here
            if new_i == len(active_list) - 1:
                fw_map[key] = ['out']
                # at current setting n_in is same for all but first layer
                n_in = json['n_nodes']
                break

            fw_map[key] = [str(new_i + 1)]

            # Add skips to the fw_map. If they are to long, sent them to output layer
            target = json['skip_' + str(old_i)] + new_i + 1
            if target >= len(active_list):
                fw_map[key].append('out')
            elif target > new_i + 1:
                fw_map[key].append(str(target))

            # Again, this is same for all but first layer
            n_in = json['n_nodes']

        setattr(self, 'out', nn.Linear(n_in, nout))

        # fw_scheme is a dict containing to which layers each layer is sending its output
        # This will fail if we have non-forward connections
        self.fw_scheme = fw_map
        print(self.fw_scheme)

    def forward(self, x):
        k = 0
        X = dict()
        X[str(k)] = [x]
        while hasattr(self, str(k)):
            # pass trough all layers except the output layer
            key = str(k)

            # we might want to concat instead of sum, then we need to modify input_size in __init__
            temp_x = sum(X[key])
            temp_out = torch.tanh(getattr(self, key)(temp_x))
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
            temp_x = x

        # Identity as output function since we do regression
        # Add support for other types od problems problems
        out = getattr(self, 'out')(temp_x)
        return out


def NAS_function(X):
    """
    Compute the branin function.
    :param X: dictionary containing the input points.
    :return: the value of the branin function
    """
    nin = 1
    nout = 1

    eps = 200    # 5000
    noise = 0.3
    my_net = json2pheno(X, nin, nout)

    loss, t = trainer(my_net, eps, noise, objective='sinx')
    score = loss * t
    # do not consider time for now
    return 100 * loss


def main():
    matplotlib.use('TkAgg')

    plt.figure(figsize=(10, 4))
    u = np.array([1, 2, 3])
    v = np.array([1, 4, 9])
    plt.scatter(u, v, color="orange")

    plt.title('Regression Analysis')

    plt.show()
    print('now we should have plots')
    parameters_file = "example_scenarios/quick_start/nas_scenario.json"
    hypermapper.optimize(parameters_file, NAS_function)
    print("End of NAS.")


if __name__ == "__main__":
    main()
