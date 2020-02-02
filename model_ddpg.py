import torch
from torch import nn
from config import args
import math
from collections import defaultdict
from torch.nn.utils import spectral_norm

action_space = args.action_space
delta = 10 # quantization levels / 2

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


class RobustNormalizer2(object):

    def __init__(self, outlier=0.1, lr=0.1):
        self.outlier = outlier
        self.lr = lr
        self.eps = 1e-5*torch.cuda.FloatTensor([1])
        self.squash_eps = 1e-5
        self.m = None
        self.n = None
        self.mu = None
        self.sigma = None

        self.y1 = -1
        self.y2 = 1

    def reset(self):
        self.m = None
        self.n = None

    def squash_derivative(self, x):
        return x

    def squash(self, x):
        x = x * self.m + self.n

        x_clamp_up = torch.clamp(x, min=self.squash_eps)
        x_clamp_down = torch.clamp(x, max=-self.squash_eps)

        x_log_up = torch.log(x_clamp_up) + 1
        x_log_down = -torch.log(-x_clamp_down) - 1

        x = x_log_up * (x >= 1).float() + x * (x >= -1).float() * (x < 1).float() + x_log_down * (x < -1).float()

        return x

    def desquash(self, x):

        x_clamp_up = torch.clamp(x, min=self.squash_eps)
        x_clamp_down = torch.clamp(x, max=-self.squash_eps)

        x_exp_up = torch.exp(x_clamp_up - 1)
        x_exp_down = -torch.exp(-(x_clamp_down + 1))

        x = x_exp_up * (x >= 1).float() + x * (x >= -1).float() * (x < 1).float() + x_exp_down * (x < -1).float()

        x = (x - self.n) / self.m
        return x

    def __call__(self, x, training=False):
        if training:
            n = len(x)
            outlier = int(n * self.outlier + .5)

            x_cpu = x.cpu()

            x2_ind = torch.kthvalue(x_cpu, n - outlier, dim=0)[1]
            x1_ind = torch.kthvalue(x_cpu, outlier, dim=0)[1]
            # curr_mu = torch.median(x_cpu, dim=0)[1]

            m = (self.y2 - self.y1) / (x[x2_ind] - x[x1_ind] + self.eps)
            n = self.y2 - m * x[x2_ind]

            if self.m is None or self.n is None:
                self.m = m
                self.n = n
            else:
                self.m = (1 - self.lr) * self.m + self.lr * m
                self.n = (1 - self.lr) * self.n + self.lr * n

            self.mu = - self.n / (self.m + self.eps)
            self.sigma = max(1 / self.m, self.eps)

        else:
            x = self.squash(x)
            return x

class NoRobustNormalizer(object):

    def __init__(self):
        self.mu = torch.zeros(1)
        self.sigma = torch.ones(1)

    def reset(self):
        return

    def squash_derivative(self, x):
        return x

    def squash(self, x):
        return x

    def desquash(self, x):
        return x

    def __call__(self, x, training=False):
        if training:
            return
        else:
            return x

class RobustNormalizer(object):

    def __init__(self, outlier=0.1, delta=1, lr=0.1):
        self.outlier = outlier
        self.delta = delta
        self.lr = lr
        self.temp_squash = nn.Tanh()
        self.eps = 1e-5*torch.cuda.FloatTensor([1])
        self.squash_eps = 1e-9
        self.mu = None
        self.sigma = None
        self.alpha = 0.1

        if args.trust_alg == 'relu':
            self.squash = self.squash_relu
            self.desquash = self.desquash_relu
            self.squash_derivative = self.squash_derivative_relu
        elif args.trust_alg == 'tanh':
            self.squash = self.squash_tanh
            self.desquash = self.desquash_tanh
            self.squash_derivative = self.squash_derivative_tanh

    def reset(self):
        self.mu = None
        self.sigma = None

    def squash_tanh(self, x):
        x = (x - self.mu) / (self.sigma + self.eps)
        x = torch.tanh(x) * (x >= 0).float() + x * (x < 0).float()
        return x

    def squash_derivative_tanh(self, x):
        x = (1 - torch.tanh(x)**2) * (x >= 0).float() + 1 * (x < 0).float()
        return self.sigma*x

    def desquash_tanh(self, x):
        x_clamp = torch.clamp(x, -1 + self.squash_eps, 1 + self.squash_eps)
        arc_tanh = 0.5 * (torch.log(1 + x_clamp) - torch.log(1 - x_clamp))
        x = arc_tanh * (x >= 0).float() + x * (x < 0).float()
        x = x * (self.sigma + self.eps) + self.mu
        return x

    def squash_relu(self, x):
        x = (x - self.mu) / (self.sigma + self.eps)
        x = (self.alpha * x + 1 - self.alpha) * (x >= 1).float() + x * (x < 1).float()
        return x

    def squash_derivative_relu(self, x):
        x = self.alpha * (x >= 0).float() + 1 * (x < 0).float()
        return self.sigma*x

    def desquash_relu(self, x):
        x = ((x - 1 + self.alpha)/self.alpha) * (x >= 1).float() + x * (x < 1).float()
        x = x * (self.sigma + self.eps) + self.mu
        return x

    def __call__(self, x, training=False):
        if training:
            n = len(x)
            outlier = int(n * self.outlier + .5)

            x_cpu = x.cpu()
            up = torch.kthvalue(x_cpu, n - outlier, dim=0)[1]
            down = torch.kthvalue(x_cpu, outlier + 1, dim=0)[1]

            mu = torch.median(x, dim=0)[0]
            sigma = (x[up] - x[down]) * self.delta

            if self.mu is None or self.sigma is None:
                self.mu = mu
                self.sigma = sigma
            else:
                self.mu = (1 - self.lr) * self.mu + self.lr * mu
                self.sigma = (1 - self.lr) * self.sigma + self.lr * sigma

            self.sigma = max(self.sigma, self.eps)

        else:
            x = self.squash(x)
            return x

class TrustRegion(object):

    def __init__(self, pi_net):
        self.mu = torch.zeros_like(pi_net.pi)
        self.sigma = torch.ones_like(pi_net.pi)
        self.pi_net = pi_net
        self.min_sigma = 0.1*torch.ones_like(pi_net.pi)
        self.trust_factor = args.trust_factor

    def bounderies(self):
        lower, upper = (self.mu - self.sigma), (self.mu + self.sigma)
        assert lower.min().item() >= -1, "lower min {}".format(lower.min())
        assert upper.max().item() <= 1, "upper max {}".format(upper.max())
        return lower, upper

    def squeeze(self, pi):

        for i in range(len(pi)):
            if pi[i] < self.mu[i] + (1 - self.min_sigma[i]) * self.sigma[i] or pi[i] > self.mu[i] - (1 - self.min_sigma[i]) * self.sigma[i]:
                self.sigma[i] = self.trust_factor * self.sigma[i]
                #print("index {}: In trust region".format(i))
            else:
                assert (self.mu[i] - self.sigma[i] >= -1), "mu - sigma < -1"
                assert (self.mu[i] + self.sigma[i] <= 1), "mu + sigma > 1"
                if self.mu[i] - self.sigma[i] == -1 or self.mu[i] + self.sigma[i] == 1:
                    self.sigma[i] = self.trust_factor * self.sigma[i]
                    print("index {}: On global boundary".format(i))
                else:
                    print("index {}: On local boundary".format(i))

        self.mu = pi

        a = torch.max(self.mu - self.sigma, -1 * torch.ones_like(self.mu))
        b = a + 2 * self.sigma
        b = torch.min(b, torch.ones_like(self.mu))
        a = b - 2 * self.sigma
        self.mu = (a + b) / 2
        self.sigma = (b - a) / 2

    def unconstrained_to_real(self, x):
        x = self.pi_net(x)
        x = self.mu + x * self.sigma
        x = torch.clamp(x, min=-1, max=1)
        return x

    def real_to_unconstrained(self, x):
        s = x.shape
        x = (x - self.mu)/self.sigma.view(1, -1)
        x = self.pi_net.inverse(x)
        x = x.view(s)
        return x

    def derivative_unconstrained(self, x):
        s = x.shape
        x = self.sigma*(1 - self.pi_net(x)**2)
        x = x.view(s)
        return x


class NoTrustRegion(object):

    def __init__(self, pi_net):
        self.mu = torch.zeros_like(pi_net.pi)
        self.sigma = torch.ones_like(pi_net.pi)
        self.pi_net = pi_net

    def bounderies(self):
        lower, upper = (self.mu - self.sigma), (self.mu + self.sigma)
        assert lower.min().item() >= -1, "lower min {}".format(lower.min())
        assert upper.max().item() <= 1, "upper max {}".format(upper.max())
        return lower, upper

    def squeeze(self, pi):
        return

    def unconstrained_to_real(self, x):
        x = self.pi_net(x)
        x = self.mu + x * self.sigma
        x = torch.clamp(x, min=-1, max=1)
        return x

    def real_to_unconstrained(self, x):
        s = x.shape
        x = (x - self.mu)/self.sigma.view(1, -1)
        x = self.pi_net.inverse(x)
        x = x.view(s)
        return x

    def derivative_unconstrained(self, x):
        s = x.shape
        x = self.sigma*(1 - self.pi_net(x)**2)
        x = x.view(s)
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
        self.output = output

    def forward(self, x, normalize=True):
        x = x.view(-1, action_space)
        if normalize:
            x = self.pi_net(x)

        x = torch.clamp(x, max=1-1e-3)

        x_emb = self.embedding(x)
        x = self.head(x, x_emb)

        return x

class SplineEmbedding(nn.Module):

    def __init__(self, device):
        super(SplineEmbedding, self).__init__()

        self.delta = delta
        self.actions = action_space
        self.emb = 32
        self.device = device

        self.ind_offset = torch.arange(self.actions, dtype=torch.int64).to(device).unsqueeze(0)

        self.b = nn.Embedding((2 * self.delta + 1) * self.actions, self.emb, sparse=True)

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

        delta = 1 / self.delta

        x = x.unsqueeze(2)
        xl = xl.unsqueeze(2)
        xh = xh.unsqueeze(2)

        h = bh / delta * (x - xl) + bl / delta * (xh - x)
        return h

class SplineHead(nn.Module):

    def __init__(self, output=1):
        super(SplineHead, self).__init__()

        self.emb = 32
        self.actions = action_space
        self.output = output
        self.global_interaction = GlobalModule(self.emb)
        self.layer = args.layer
        input_len = self.emb + self.actions

        self.fc = nn.Sequential(nn.Linear(input_len, self.layer, bias=True),
                                ResBlock(self.layer),
                                ResBlock(self.layer),
                                nn.ReLU(),
                                nn.Linear(self.layer, output, bias=True))

        init_weights(self, init='ortho')

    def forward(self, x, x_emb):
        h = x_emb.transpose(2, 1)
        h = self.global_interaction(h)

        x = torch.cat([x, h], dim=1)

        x = self.fc(x)
        if self.output == 1:
            x = x.squeeze(1)

        return x

class ResBlock(nn.Module):

    def __init__(self, layer):

        super(ResBlock, self).__init__()
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
        self.emb = 32

        self.blocks = nn.Sequential(
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
        self.emb = 32

        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True),
        )

        self.key = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True),
        )

        self.value = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=True),
        )

        self.output = nn.Sequential(
            nn.ReLU(),
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
        policy = torch.clamp(policy, min=-1 + 1e-5, max=1 - 1e-5)
        return 0.5 * (torch.log(1 + policy) - torch.log(1 - policy))

    def inverse_derivative(self,  policy):
        return 0.5 / (1 + policy) - 1 / (1 - policy)

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
