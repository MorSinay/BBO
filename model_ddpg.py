import torch
from torch import nn
from config import args
import torch.nn.functional as F


action_space = args.action_space
layer = args.layer
drop = args.drop

class DuelNet(nn.Module):

    def __init__(self):

        super(DuelNet, self).__init__()

        layer = args.layer

        self.fc = nn.Sequential(nn.Linear(action_space, 2*layer, bias=False),
                                nn.BatchNorm1d(2*layer),
                                nn.LeakyReLU(),
                                nn.Dropout(drop),
                                nn.Linear(2*layer, 2*layer, bias=False),
                                nn.BatchNorm1d(2*layer),
                                nn.LeakyReLU(),
                                nn.Dropout(drop),
                                nn.Linear(2*layer, layer, bias=False),
                                nn.BatchNorm1d(layer),
                                nn.LeakyReLU(),
                                nn.Dropout(drop),
                                nn.Linear(layer, 1))

    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, pi):
        pi = pi.view(-1, action_space)
        x = self.fc(pi)

        return x


class ResNet(nn.Module):

    def __init__(self):

        super(ResNet, self).__init__()

        layer = args.layer

        self.fc = nn.Sequential(nn.Linear(action_space, 2*layer, bias=False),
                                nn.BatchNorm1d(2*layer),
                                nn.LeakyReLU(),
                                nn.Dropout(drop),
                                ResBlock(din=2*layer, expansion=1, drop=drop),
                                ResBlock(din=2*layer, expansion=0.5, drop=drop),
                                nn.Linear(layer, 1))


    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, pi):
        pi = pi.view(-1, action_space)
        x = self.fc(pi)

        return x


class ResBlock(nn.Module):

    def __init__(self, din=layer, expansion=1, drop=0.1):

        super(ResBlock, self).__init__()

        self.expansion = expansion

        self.fc = nn.Sequential(nn.Linear(din, din, bias=False),
                                nn.BatchNorm1d(din),
                                nn.LeakyReLU(),
                                nn.Dropout(drop),
                                nn.Linear(din, int(expansion*din), bias=False),
                                nn.BatchNorm1d(int(expansion*din)),
                                nn.LeakyReLU(),
                                nn.Dropout(drop))

    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, x):
        if self.expansion >= 1:
            x0 = x.repeat(1, self.expansion)
        else:
            down_sample = int(1/self.expansion)
            din = x.shape[-1]
            x0 = x.view(-1, din // down_sample, down_sample)
            x0 = x0.mean(dim=2)

        x = self.fc(x)

        return x + x0

