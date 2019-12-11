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

        self.fc = nn.Sequential(nn.Tanh(),
                                nn.Linear(action_space, 2*layer, bias=False),
                                #nn.BatchNorm1d(2*layer),
                                nn.LeakyReLU(),
                                nn.Dropout(drop),
                                nn.Linear(2*layer, 2*layer, bias=False),
                                #nn.BatchNorm1d(2*layer),
                                nn.LeakyReLU(),
                                nn.Dropout(drop),
                                nn.Linear(2*layer, layer, bias=False),
                                #nn.BatchNorm1d(layer),
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

class PiNet(nn.Module):

    def __init__(self, init, device, action_space):

        super(PiNet, self).__init__()

        self.pi = nn.Parameter(init)
        self.normalize = nn.Tanh()
        self.device = device
        self.action_space = action_space

    def forward(self, pi=None):
        if pi is None:
            return self.normalize(self.pi)
        else:
            return self.normalize(pi)

    def pi_update(self, pi):
        with torch.no_grad():
            self.pi.data = pi

    def grad_update(self, grads):
    #    if self.action_space == 1:
     #       grads = grads.squeeze(0)
        with torch.no_grad():
            self.pi.grad = grads

class PiNetClamp(nn.Module):

    def __init__(self, init, device, action_space):

        super(PiNet, self).__init__()

        self.pi = nn.Parameter(init)
        self.device = device
        self.action_space = action_space

    def forward(self, pi=None):
        if pi is None:
            return torch.clamp(self.pi, -1, 1)
        else:
            return torch.clamp(pi, -1, 1)

    def pi_update(self, pi):
        with torch.no_grad():
            self.pi.data = torch.clamp(pi)

    def grad_update(self, grads):
        with torch.no_grad():
            self.pi.grad = grads


class DerivativeNet(nn.Module):

    def __init__(self):

        super(DerivativeNet, self).__init__()

        layer = args.layer

        self.fc = nn.Sequential(nn.Tanh(),
                                nn.Linear(action_space, 2*layer, bias=False),
                                #nn.BatchNorm1d(2*layer),
                                nn.LeakyReLU(),
                                nn.Dropout(drop),
                                nn.Linear(2*layer, 2*layer, bias=False),
                                #nn.BatchNorm1d(2*layer),
                                nn.LeakyReLU(),
                                nn.Dropout(drop),
                                nn.Linear(2*layer, layer, bias=False),
                                #nn.BatchNorm1d(layer),
                                nn.LeakyReLU(),
                                nn.Dropout(drop),
                                nn.Linear(layer, action_space))

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

