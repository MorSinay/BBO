import torch
from torch import nn
from config import args
from torch.nn.utils import spectral_norm
import math
from collections import defaultdict


action_space = args.action_space
layer = 512
drop = args.drop
delta = 20 # quantization levels / 2
hn = 10
emb = 64 # embedding size
parallel = int(layer / emb + .5)
emb2 = int(layer / action_space + .5) # embedding2 size

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

    def __init__(self, pi):
        self.mu = pi
        self.sigma = torch.ones_like(pi)

    def squeeze(self, pi):
        self.mu = pi
        self.sigma /= 2

    def __call__(self, x):
        x = self.mu + x * self.sigma
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
        for op in self.optimizers:
            op_dict[op.__name__] = op.state_dict()

    def load_state_dict(self, op_dict):
        for op in self.optimizers:
            op.load_state_dict(op_dict[op.__name__])

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
        x_emb, x_emb2 = self.embedding(x)
        x = self.head(x, x_emb, x_emb2)

        return x

class SplineEmbedding(nn.Module):

    def __init__(self, device):
        super(SplineEmbedding, self).__init__()

        self.delta = delta
        self.actions = action_space
        self.emb = emb
        self.emb2 = emb2
        self.device = device

        self.a = nn.Embedding((2 * self.delta + 1) * self.actions, emb, sparse=True)
        self.b = nn.Embedding((2 * self.delta + 1) * self.actions, emb, sparse=True)

        self.a2 = nn.Embedding((2 * self.delta + 1) * self.actions, emb2, sparse=True)
        self.b2 = nn.Embedding((2 * self.delta + 1) * self.actions, emb2, sparse=True)

        nn.init.zeros_(self.a.weight)
        nn.init.zeros_(self.a2.weight)

    def forward(self, x):
        xl = (x * self.delta).floor()
        xli = xl.long() + self.delta
        xl = xl / self.delta

        xh = (x * self.delta).ceil()
        xhi = xh.long() + self.delta
        xh = xh / self.delta

        n = len(xhi)
        xhi = xhi.view(-1)
        xli = xli.view(-1)

        ind_offset = (2 * self.delta + 1) * torch.arange(self.actions).to(self.device).repeat(n)

        al = self.a(xli + ind_offset).view(n, self.actions, self.emb)
        ah = self.a(xhi + ind_offset).view(n, self.actions, self.emb)

        bl = self.b(xli + ind_offset).view(n, self.actions, self.emb)
        bh = self.b(xhi + ind_offset).view(n, self.actions, self.emb)

        al2 = self.a2(xli + ind_offset).view(n, self.actions, self.emb2)
        ah2 = self.a2(xhi + ind_offset).view(n, self.actions, self.emb2)

        bl2 = self.b2(xli + ind_offset).view(n, self.actions, self.emb2)
        bh2 = self.b2(xhi + ind_offset).view(n, self.actions, self.emb2)

        delta = 1 / self.delta

        x = x.unsqueeze(2)
        xl = xl.unsqueeze(2)
        xh = xh.unsqueeze(2)

        h = ah * (x - xl) ** 3 / (6 * delta) + al * (xh - x) ** 3 / (6 * delta) + \
            (bh / delta - ah * delta / 6) * (x - xl) + (bl / delta - al / 6) * (xh - x)

        h2 = ah2 * (x - xl) ** 3 / (6 * delta) + al2 * (xh - x) ** 3 / (6 * delta) + \
             (bh2 / delta - ah2 * delta / 6) * (x - xl) + (bl2 / delta - al2 / 6) * (xh - x)

        return h, h2

class SplineHead(nn.Module):

    def __init__(self, output=1):
        super(SplineHead, self).__init__()

        self.emb = emb
        self.actions = action_space
        self.output = output
        self.emb2 = emb2
        self.global_interaction = nn.ModuleList([GlobalModule(emb) for _ in range(parallel)])

        input_len = parallel * emb + self.actions + emb2 * self.actions

        self.fc = nn.Sequential(spectral_norm(nn.Linear(input_len, layer, bias=False)),
                                ResBlock(layer),
                                ResBlock(layer),
                             #   ResBlock(layer),
                              #  ResBlock(layer),
                               # ResBlock(layer),
                                ResBlock(layer),
                                nn.BatchNorm1d(layer, affine=True),
                                nn.ReLU(),
                                spectral_norm(nn.Linear(layer, output, bias=True)))

        init_weights(self, init='ortho')

    def forward(self, x, x_emb, x_emb2):
        h2 = x_emb2.view(len(x_emb2), self.emb2 * self.actions)

        h = x_emb.transpose(2, 1)
        h = torch.cat([gi(h) for gi in self.global_interaction], dim=1)

        #x = x.squeeze(2)

        x = torch.cat([x, h, h2], dim=1)

        x = self.fc(x)
        if self.output == 1:
            x = x.squeeze(1)

        return x

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
            nn.BatchNorm1d(planes, affine=True),
            nn.ReLU(),
            spectral_norm(nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False)),
        )

        self.key = nn.Sequential(
            nn.BatchNorm1d(planes, affine=True),
            nn.ReLU(),
            spectral_norm(nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False)),
        )

        self.value = nn.Sequential(
            nn.BatchNorm1d(planes, affine=True),
            nn.ReLU(),
            spectral_norm(nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False)),
        )

        self.output = nn.Sequential(
            nn.BatchNorm1d(planes, affine=True),
            nn.ReLU(),
            spectral_norm(nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False)),
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

    def __init__(self, pi_net):

        super(DuelNet, self).__init__()
        self.pi_net = pi_net
        layer = args.layer

        self.fc = nn.Sequential(nn.Linear(action_space, layer, bias=True),
                                nn.ReLU(),
                                nn.Linear(layer, 2*layer, bias=True),
                                #nn.BatchNorm1d(2*layer),
                                nn.ReLU(),
                                #nn.Dropout(drop),
                                nn.Linear(2*layer, 2*layer, bias=True),
                                #nn.BatchNorm1d(2*layer),
                                nn.ReLU(),
                                #nn.Dropout(drop),
                                nn.Linear(2*layer, layer, bias=True),
                                #nn.BatchNorm1d(layer),
                                nn.ReLU(),
                                #nn.Dropout(drop),
                                nn.Linear(layer, 1))

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

class DerivativeNet(nn.Module):

    def __init__(self, pi_net):

        super(DerivativeNet, self).__init__()
        self.pi_net = pi_net
        layer = args.layer

        self.fc = nn.Sequential(nn.Linear(action_space, 2*layer, bias=True),
                                #nn.BatchNorm1d(2*layer),
                                nn.ReLU(),
                                #nn.Dropout(drop),
                                nn.Linear(2*layer, 2*layer, bias=True),
                                #nn.BatchNorm1d(2*layer),
                                nn.ReLU(),
                                #nn.Dropout(drop),
                                nn.Linear(2*layer, layer, bias=True),
                                #nn.BatchNorm1d(layer),
                                nn.ReLU(),
                                #nn.Dropout(drop),
                                nn.Linear(layer, action_space))

    def reset(self):
        for weight in self.parameters():
            nn.init.xavier_uniform(weight.data)

    def forward(self, pi, normalize=True):
        pi = pi.view(-1, action_space)
        if normalize:
            pi = self.pi_net(pi)
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

