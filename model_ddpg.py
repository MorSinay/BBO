import torch
from torch import nn
from config import args
#from torch.nn.utils import spectral_norm
import math
from collections import defaultdict
import numpy as np

action_space = args.action_space
layer = 256
delta = 10 # quantization levels / 2
emb = 32 # embedding size
#parallel = 2 #int(layer / emb + .5)
#emb2 = int(layer / action_space + .5) # embedding2 size

def init_weights(net, init='ortho'):
    net.param_count = 0
    for module in net.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear,
                               nn.ConvTranspose2d, nn.ConvTranspose1d)):
            if init == 'ortho':
                torch.nn.init.orthogonal_(module.weight)
            elif init == 'N02':
                torch.nn.init.normal_(module.weight, 0, 0.02)
            elif init in ['glorot', 'xavier']:
                torch.nn.init.xavier_uniform_(module.weight)
            else:
                print('Init style not recognized...')

        net.param_count += sum([p.data.nelement() for p in module.parameters()])


class RobustNormalizer(object):

    def __init__(self, outlayer=0.1, delta=1, lr=0.1):
        self.outlayer = outlayer
        self.delta = delta
        self.lr = lr
        self.squash = nn.Tanh()
        self.eps = 1e-5
        self.squash_eps = 1e-9
        self.mu = None
        self.sigma = None

    def reset(self):
        self.mu = None
        self.sigma = None

    def desquash(self, x):

        x = torch.clamp(x, -1 + self.squash_eps, 1 + self.squash_eps)
        return 0.5 * (torch.log(1 + x) - torch.log(1 - x))

    def __call__(self, x, training=False):
        if training:
            n = len(x)
            outlayer = int(n * self.outlayer + .5)
            up = torch.kthvalue(x, n - outlayer, dim=0)[0]
            down = torch.kthvalue(x, outlayer + 1, dim=0)[0]

            mu = torch.median(x, dim=0)[0]
            sigma = (up - down) * self.delta

            if self.mu is None or self.sigma is None:
                self.mu = mu
                self.sigma = sigma
            else:
                self.mu = (1 - self.lr) * self.mu + self.lr * mu
                self.sigma = (1 - self.lr) * self.sigma + self.lr * sigma

        else:
            x = self.squash((x - self.mu) / (self.sigma + self.eps))
            return x

class TrustRegion(object):

    def __init__(self, pi_net):
        self.mu = torch.zeros_like(pi_net.pi).cpu()
        self.sigma = 1.
        self.pi_net = pi_net
        self.min_sigma = 0.1
#        self.a = -1*torch.ones_like(pi_net.pi).cpu()
#        self.b = torch.ones_like(pi_net.pi).cpu()

    def np_bounderies(self):
        lower, upper = (self.mu - self.sigma).numpy(), (self.mu + self.sigma).numpy()
        # assert lower.min() >= -1, "lower min {}".format(lower.min())
        # assert upper.max() <= 1, "upper max {}".format(upper.max())
        return np.clip(lower, -1, 1), np.clip(upper, -1, 1)
        #return lower, upper

    def squeeze(self, pi):
        lower, upper = self.np_bounderies()
        lower, upper = lower + self.min_sigma/2, upper - self.min_sigma/2
        new_pi = np.clip(pi.numpy(), a_max=upper, a_min=lower)
        #near the edge
        if (new_pi == pi.numpy()).min(): # or pi.min() == -1 or pi.max == 1:
            self.sigma = max(3*self.sigma/4, self.min_sigma)
        else:
            print("not in range - squeeze")

        self.mu = pi
        #self.a = torch.max(self.mu - self.sigma, -1 * torch.ones_like(self.mu))
        #self.b = torch.min(self.mu + self.sigma, torch.ones_like(self.mu))
        #self.mu = torch.FloatTensor((self.a + self.b) / 2)
        #self.sigma = torch.FloatTensor((self.b - self.a) / 2)

    def unconstrained_to_real(self, x):
        x = self.pi_net(x)
        x = self.mu + x * self.sigma
        x = torch.clamp(x, min=-1, max=1)
        return x

    def real_to_unconstrained(self, x):
        x = (x - self.mu)/self.sigma
        x = self.pi_net.inverse(x)
        return x

class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def state_dict(self):
        op_dict = defaultdict()
        for i, op in enumerate(self.optimizers):
            op_dict[str(i)] = op.state_dict()

    def load_state_dict(self, op_dict):
        for i, op in enumerate(self.optimizers):
            op.load_state_dict(op_dict[str(i)])

class SplineNet(nn.Module):

    def __init__(self, device, pi_net, output=1):
        super(SplineNet, self).__init__()
        self.pi_net = pi_net
        self.embedding = SplineEmbedding(device)
        self.head = SplineHead(output)

    def forward(self, x, normalize=True):
        x = x.view(-1, action_space)
        if normalize:
            x = self.pi_net(x)
        # x_emb, x_emb2 = self.embedding(x)
        # x = self.head(x, x_emb, x_emb2)

        x_emb = self.embedding(x)
        x = self.head(x, x_emb)

        return x

class SplineEmbedding(nn.Module):

    def __init__(self, device):
        super(SplineEmbedding, self).__init__()

        self.delta = delta
        self.actions = action_space
        self.emb = emb
        #self.emb2 = emb2
        self.device = device

        self.ind_offset = torch.arange(self.actions, dtype=torch.int64).to(device).unsqueeze(0)

        self.b = nn.Embedding((2 * self.delta + 1) * self.actions, emb, sparse=True)
        #self.b2 = nn.Embedding((2 * self.delta + 1) * self.actions, emb2, sparse=True)

    def forward(self, x):
        n = len(x)

        xl = (x * self.delta).floor()
        xli = self.actions * (xl.long() + self.delta) + self.ind_offset
        xl = xl / self.delta
        xli = xli.view(-1)


        xh = (x * self.delta + 1).floor()
        xhi = self.actions * (xh.long() + self.delta) + self.ind_offset
        xh = xh / self.delta
        xhi = xhi.view(-1)

        bl = self.b(xli).view(n, self.actions, self.emb)
        bh = self.b(xhi).view(n, self.actions, self.emb)

       # bl2 = self.b2(xli).view(n, self.actions, self.emb2)
       # bh2 = self.b2(xhi).view(n, self.actions, self.emb2)

        delta = 1 / self.delta

        x = x.unsqueeze(2)
        xl = xl.unsqueeze(2)
        xh = xh.unsqueeze(2)

        h = bh / delta * (x - xl) + bl / delta * (xh - x)
       # h2 = bh2 / delta* (x - xl) + bl2 / delta * (xh - x)

       # return h, h2
        return h

class SplineHead(nn.Module):

    def __init__(self, output=1):
        super(SplineHead, self).__init__()

        self.emb = emb
        self.actions = action_space
        self.output = output
       # self.emb2 = emb2
        self.global_interaction = GlobalModule(emb) #nn.ModuleList([GlobalModule(emb) for _ in range(parallel)])

        #input_len = parallel * emb + self.actions #+ emb2 * self.actions
        input_len = emb + self.actions #+ emb2 * self.actions

        self.fc = nn.Sequential(#spectral_norm(nn.Linear(input_len, layer, bias=False)),
                                nn.Linear(input_len, layer, bias=True),
                                ResBlock(layer),
                                ResBlock(layer),
                             #   ResBlock(layer),
                              #  ResBlock(layer),
                               # ResBlock(layer),
                               # ResBlock(layer),
                                #nn.BatchNorm1d(layer, affine=True),
                                nn.ReLU(),
                                #spectral_norm(nn.Linear(layer, output, bias=True)))
                                nn.Linear(layer, output, bias=True))

        init_weights(self, init='ortho')

    #def forward(self, x, x_emb, x_emb2):
    def forward(self, x, x_emb):
        #h2 = x_emb2.view(len(x_emb2), self.emb2 * self.actions)

        h = x_emb.transpose(2, 1)
        #h = torch.cat([gi(h) for gi in self.global_interaction], dim=1)
        h = self.global_interaction(h)

#        x = x.squeeze(2)

        #x = torch.cat([x, h, h2], dim=1)
        x = torch.cat([x, h], dim=1)

        x = self.fc(x)
        if self.output == 1:
            x = x.squeeze(1)

        return x

class ResBlock(nn.Module):

    def __init__(self, layer):

        super(ResBlock, self).__init__()

        # self.fc = nn.Sequential(nn.BatchNorm1d(layer, affine=True),
        #                         nn.ReLU(),
        #                         spectral_norm(nn.Linear(layer, layer, bias=False)),
        #                         nn.BatchNorm1d(layer, affine=True),
        #                         nn.ReLU(),
        #                         spectral_norm(nn.Linear(layer, layer, bias=False)),
        #                        )

        self.fc = nn.Sequential(nn.ReLU(),
                                nn.Linear(layer, layer, bias=True),
                                nn.ReLU(),
                                nn.Linear(layer, layer, bias=True),
                               )

    def forward(self, x):

        h = self.fc(x)
        return x + h

class GlobalModule(nn.Module):

    def __init__(self, planes):
        super(GlobalModule, self).__init__()

        self.actions = action_space
        self.emb = emb

        self.blocks = nn.Sequential(
        #    GlobalBlock(planes),
            GlobalBlock(planes),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.blocks(x)
        x = x.squeeze(2)

        return x

class GlobalBlock(nn.Module):

    def __init__(self, planes):
        super(GlobalBlock, self).__init__()

        self.actions = action_space
        self.emb = emb

        self.query = nn.Sequential(
            #nn.BatchNorm1d(planes, affine=True),
            nn.ReLU(),
            #spectral_norm(nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False)),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True),
        )

        self.key = nn.Sequential(
            # nn.BatchNorm1d(planes, affine=True),
            nn.ReLU(),
            # spectral_norm(nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False)),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True),
        )

        self.value = nn.Sequential(
            # nn.BatchNorm1d(planes, affine=True),
            nn.ReLU(),
            # spectral_norm(nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False)),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True),
        )

        self.output = nn.Sequential(
            # nn.BatchNorm1d(planes, affine=True),
            nn.ReLU(),
            # spectral_norm(nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False)),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True),
        )

        self.planes = planes

    def forward(self, x):
        q = self.query(x).transpose(1, 2)
        k = self.key(x)
        v = self.value(x).transpose(1, 2)

        a = torch.softmax(torch.bmm(q, k) / math.sqrt(self.planes), dim=2)
        r = torch.bmm(a, v).transpose(1, 2)
        r = self.output(r)

        return x + r

class DuelNet(nn.Module):

    def __init__(self, pi_net, output):

        super(DuelNet, self).__init__()
        self.pi_net = pi_net
        layer = args.layer

        self.fc = nn.Sequential(nn.Linear(action_space, layer, bias=True),
                                nn.ReLU(),
                                nn.Linear(layer, 2*layer, bias=True),
                                nn.ReLU(),
                                nn.Linear(2*layer, 2*layer, bias=True),
                                nn.ReLU(),
                                nn.Linear(2*layer, layer, bias=True),
                                nn.ReLU(),
                                nn.Linear(layer, output))

    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, pi, normalize=True):
        pi = pi.view(-1, action_space)
        if normalize:
            pi = self.pi_net(pi)
        x = self.fc(pi)

        return x

class PiNet(nn.Module):

    def __init__(self, init, device, action_space):

        super(PiNet, self).__init__()
        self.pi = nn.Parameter(init)
        self.normalize = nn.Tanh()
        self.device = device
        self.action_space = action_space

    def inverse(self,  policy):
        return 0.5 * (torch.log(1 + policy) - torch.log(1 - policy))

    def forward(self, pi=None):
        if pi is None:
            return self.normalize(self.pi)
        else:
            return self.normalize(pi)

    def pi_update(self, pi):
        with torch.no_grad():
            self.pi.data = pi

    def grad_update(self, grads):
        with torch.no_grad():
            self.pi.grad = grads
