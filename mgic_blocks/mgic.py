#!/usr/bin/env python3

import torch
import torch.nn as nn
import math


class Restr2d(nn.Module):
    def __init__(self, in_channels: int, sg: int):
        super(Restr2d, self).__init__()

        self.in_channels = in_channels
        self.sg = sg
        self.kernel = (1, 1)
        if self.in_channels % self.sg != 0:
            raise Exception("Number of groups not compatible with number of input channels")

        self.out_channels = self.in_channels // self.sg
        self._restriction = nn.Conv2d(self.in_channels, self.out_channels, self.kernel, groups=self.sg)

    def forward(self, x):
        out = self._restriction(x)
        return out


class Prol2d(nn.Module):
    def __init__(self, in_channels: int, sg: int):
        super(Prol2d, self).__init__()

        self.in_channels = in_channels
        self.sg = sg
        self.out_channels = self.in_channels * self.sg
        self.kernel = (1, 1)
        self.prolongation = nn.Conv2d(self.in_channels, self.out_channels, self.kernel, groups=self.sg)

    def forward(self, x: torch.Tensor):
        out = self.prolongation(x)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, seq: tuple, kernel=(3, 3)):
        super(BasicBlock, self).__init__()
        self.cin = in_channels
        self.sg = seq[0]
        self.sc = seq[-1]
        self.kernel = kernel
        self.n_levels = len(seq)

        self.res_seq = []
        self.conv_seq = []
        self.prol_seq = []
        self.bn_seq = []
        level_channels = [int(self.cin // math.prod(seq[:level]))
                          for level in range(self.n_levels + 1)]

        for level in range(self.n_levels):
            self.res_seq.append(Restr2d(level_channels[level], seq[level]))
            self.prol_seq.append(Prol2d(level_channels[level + 1], seq[level]))

        for level in range(self.n_levels - 1, -1, -1):
            self.conv_seq.append(nn.Conv2d(level_channels[level], level_channels[level],
                                           self.kernel, padding=1, groups=seq[level]))
            self.bn_seq.append(nn.BatchNorm2d(level_channels[level]))

        self.convc = nn.Conv2d(level_channels[-1], level_channels[-1], self.kernel, padding=1)

        self.res_seq = nn.Sequential(*self.res_seq)
        self.conv_seq = nn.Sequential(*self.conv_seq)
        self.prol_seq = nn.Sequential(*self.prol_seq)
        self.bn_seq = nn.Sequential(*self.bn_seq)

    def forward(self, x: torch.Tensor):
        x_seq = [x]
        temp = x.clone()
        for layer in self.res_seq:
            temp = layer(temp)
            x_seq.append(temp)
        temp = self.convc(temp)

        for level in range(self.n_levels):
            temp = x_seq[::-1][level + 1] + self.bn_seq[level](self.prol_seq[::-1][level](temp - x_seq[::-1][level]))
            temp = self.conv_seq[level](temp)

        return temp


class Net(nn.Module):
    def __init__(self, Block, layer_list, num_classes, num_channels=3):
        super(Net, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Block, layer_list[0], planes=64)
        self.layer2 = self._make_layer(Block, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(Block, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(Block, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Block.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, Block, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * Block.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * Block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * Block.expansion)
            )

        layers.append(Block(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * Block.expansion

        for i in range(blocks - 1):
            layers.append(Block(self.in_channels, planes))

        return nn.Sequential(*layers)


def Net34(num_classes, channels=3):
    return Net(BasicBlock, [3, 4, 6, 3], num_classes, channels)
