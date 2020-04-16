import torch
import torch.nn as nn
import time

import torch.optim as optim

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomCrop, RandomHorizontalFlip
from torch.utils.data import DataLoader

def get_data_loaders(train_batch_size, test_batch_size, size=(224,224)):
    # Augments data like 4.2 in resnetpaper, is normalization right??
    transform_train = Compose([RandomHorizontalFlip(), Resize(size), RandomCrop(32, padding=4),
                               ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = Compose([Resize(size), ToTensor(),
                              Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_loader = DataLoader(CIFAR10('data/', download=False, transform=transform_train, train=True),
                              batch_size=train_batch_size, shuffle=True, num_workers=2)

    test_loader = DataLoader(CIFAR10('data/', download=False, transform=transform_test, train=False),
                             batch_size=test_batch_size, shuffle=False, num_workers=2)

    print('loaded the mnist data')
    return train_loader, test_loader

# counts trainable weights in a model
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Base for ResNet
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
        groups = 1,
        base_width = 64
        dilation = 1
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

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


class ResNet(nn.Module):
    # ResNet(BasicBlock, [2, 2, 2, 2])
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()

        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 16

        self.groups = 1
        self.base_width = 64
        # Första filtret för ResNet "börjar"
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        # Slutfas
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # What is this???
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # might want to Zero-initialize the last BN in each residual branch here

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = 1
        dilate = False

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
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # We do not need softmax since it is done inside nn.CrossEntropyLoss()
        return x

    def forward(self, x):
        return self._forward_impl(x)


def train(network, train_loader, test_loader, device, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters())

    t0 = time.perf_counter()
    network.to(device)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
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
            #running_loss += loss.item()
            #if i % 100 == 99:  # print every 20 mini-batches
             #   print('[%d, %5d] loss: %.3f' %
              #        (epoch + 1, i + 1, running_loss / 100))
               # running_loss = 0.0
        if epoch % 10 == 9:
            validate(network, test_loader, device, t0, epoch=epoch+1)

    print('Finished Training it took ', (time.perf_counter() - t0) / 60, ' minutes to train')


def validate(network, test_loader, device, t0, epoch=0):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %f %%' % (
                100 * correct / total))
    print('Have trained ', epoch, ' out of 160 epochs in ', (time.perf_counter() - t0)/3600, ' hours')


n_epochs = 160
batch_size_train = 128
batch_size_test = 1000
size = (32,32)     # ResNet is made for 224, Mnist is 28, Cifar-10 is 32
train_loader, test_loader = get_data_loaders(batch_size_train, batch_size_test, size=size)

# Call RNsmall
n = 18
print('small resnet of deepth ', 6*n + 2)

resnetGPU = ResNet(BasicBlock, [n,n,n])
print('params: ', count_params(resnetGPU))
deviceG = torch.device("cuda:0")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print('device: ', deviceG.type)
train(resnetGPU, train_loader, test_loader, deviceG, n_epochs)
"""
resnetCPU = ResNet(BasicBlock, [n,n,n,n])
print('params: ', count_params(resnetCPU))
deviceC = torch.device("cpu")
print('device: ', deviceC.type)
train(resnetCPU, train_loader, deviceC, n_epochs)
"""
