import torch
from torch import nn
from config import args
import torch.nn.functional as F

action_space = args.action_space


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

    @property
    def detach(self):
        return self.pi.detach

    @property
    def grad(self):
        return self.pi.grad

    def inverse(self,  policy):
        policy = torch.clamp(policy, min=-1 + 1e-3, max=1 - 1e-3)
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
        b = torch.min(self.mu + self.sigma, torch.ones_like(self.mu))
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


class RobustNormalizer(object):

    def __init__(self, outlier=0.1, lr=0.1):
        self.outlier = outlier
        self.lr = lr
        self.eps = 1e-5
        self.squash_eps = 1e-5
        self.m = None
        self.n = None
        self.mu = None
        self.sigma = None

        self.y1 = -2
        self.y2 = 2

    def reset(self):
        self.m = None
        self.n = None

    def squash(self, x):

        x = x * self.m + self.n

        x = torch.tanh(x) * (x >= 0).float() + x * (x < 0).float()

        return x

    def __call__(self, x, training=False):
        if training:
            n = len(x)
            outlier = int(n * self.outlier + .5)

            x_cpu = x.cpu()

            x2_ind = torch.kthvalue(x_cpu, n - outlier, dim=0)[1]
            x1_ind = torch.kthvalue(x_cpu, outlier, dim=0)[1]
            # curr_mu = torch.median(x_cpu, dim=0)[1]

            m = (self.y2 - self.y1) / (x[x2_ind] - x[x1_ind])
            n = self.y2 - m * x[x2_ind]

            if self.m is None or self.n is None:
                self.m = m
                self.n = n
            else:
                self.m = (1 - self.lr) * self.m + self.lr * m
                self.n = (1 - self.lr) * self.n + self.lr * n

            self.mu = - self.n / self.m
            self.sigma = 1 / self.m

        else:
            x = self.squash(x)
            return x


# ADD these classes in order to load pretrained model


from torch import nn
import torch
import torch.nn.functional as F
from config import args, exp
from torch.autograd import Function
from collections import namedtuple
from torchvision import transforms
from torch.nn.utils import spectral_norm
import math

float_func = torch.cuda.HalfTensor if args.half else torch.cuda.FloatTensor

LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])


# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(object):

    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.module.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, x):

        x = x / 2 + 1
        x = torch.stack([self.normalize(xi) for xi in x])

        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return output['relu1_2'], output['relu2_2'], output['relu3_3'], output['relu4_3']


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        print("Reversed")
        print(grads)
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        print(dx)
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class GPFuncLayer(nn.Module):

    def __init__(self, func, *argc, **kwargs):
        super(GPFuncLayer, self).__init__()
        self.func = func
        self.argc = argc
        self.kwargs = kwargs

    def forward(self, x):
        return self.func(x, *self.argc, **self.kwargs)


class GPAttrLayer(nn.Module):

    def __init__(self, func, *argc, **kwargs):
        super(GPAttrLayer, self).__init__()
        self.func = func
        self.argc = argc
        self.kwargs = kwargs

    def forward(self, x):

        f = getattr(x, self.func)
        return f(*self.argc, **self.kwargs)


class VarLayer(nn.Module):

    def __init__(self, sample=True):
        super(VarLayer, self).__init__()
        self.sample = sample

    def forward(self, mean, std):

        if self.training or self.sample:
            mean = mean + float_func(std.shape).normal_() * std

        return mean


class ResversalGAN(nn.Module):

    def __init__(self):
        super(ResversalGAN, self).__init__()

        self.gen = Generator()
        self.rev = GradientReversal(args.lr_g / args.lr_d)
        self.disc = Discriminator()

    def freeze_discriminator(self, freeze=True):
        for param in self.disc.parameters():
            param.requires_grad = freeze

    def forward(self, z, x_real=None):
        x = self.gen(z)
        x = self.rev(x)
        y_fake = self.disc(x)

        if x_real is None:
            return x, y_fake
        else:
            y_real = self.disc(x_real)
            return x, y_fake, y_real


class SelfExpansion(nn.Module):

    def __init__(self, planes, expansion=1):
        super(SelfExpansion, self).__init__()

        self.conv = spectral_norm(nn.Conv2d(planes, expansion * planes - planes, kernel_size=1, bias=True))

    def forward(self, x):
        h = self.conv(x)
        return torch.cat([x, h], dim=1)


class UpSample(nn.Module):

    def __init__(self, upsample=1):
        super(UpSample, self).__init__()
        self.upsample = upsample

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.upsample)
        # x = F.interpolate(x, scale_factor=self.upsample, mode='bilinear', align_corners=True)
        return x


class UpSampleConv(nn.Module):

    def __init__(self, planes, upsample=1):
        super(UpSampleConv, self).__init__()

        self.eye = nn.Parameter(torch.eye(planes).view(planes, planes, 1, 1), requires_grad=False)
        self.upsample = upsample

    def forward(self, x):
        x = F.conv_transpose2d(x, self.eye, stride=self.upsample, output_padding=int(self.upsample > 1))
        return x


class ResNormDown(nn.Module):

    def __init__(self, planes, expansion=1, downsample=1):
        super(ResNormDown, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(planes, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            # nn.Dropout(args.dropout),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(planes, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            # nn.Dropout(args.dropout),
            nn.Conv2d(planes, expansion * planes, kernel_size=3, padding=1, bias=True),
            # nn.MaxPool2d(downsample),
            nn.AvgPool2d(downsample),
        )

        if expansion == 1:
            self.identity = nn.AvgPool2d(downsample)  # nn.MaxPool2d(downsample)
        else:
            self.identity = nn.Sequential(
                SelfExpansion(planes, expansion),
                nn.MaxPool2d(downsample)
            )

    def forward(self, x):
        h = self.conv(x)
        x = self.identity(x)

        return x + h


class ResDown(nn.Module):

    def __init__(self, planes, expansion=1, downsample=1):
        super(ResDown, self).__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            spectral_norm(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=True)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(planes, expansion * planes, kernel_size=3, padding=1, bias=True)),
            nn.AvgPool2d(downsample),
        )

        if expansion == 1:
            self.identity = nn.AvgPool2d(downsample)
        else:
            self.identity = nn.Sequential(
                SelfExpansion(planes, expansion),
                nn.AvgPool2d(downsample)
            )

    def forward(self, x):
        h = self.conv(x)
        x = self.identity(x)

        return x + h


class ConditionalBatchNorm(nn.Module):

    def __init__(self, planes, planes_in, track_running_stats=True, momentum=0.01):
        super(ConditionalBatchNorm, self).__init__()

        self.bn = nn.BatchNorm2d(planes, affine=False, track_running_stats=True, momentum=0.01)
        self.lin_gamma = spectral_norm(nn.Linear(planes_in, planes, bias=False))
        self.lin_beta = spectral_norm(nn.Linear(planes_in, planes, bias=False))
        self.planes = planes

    def forward(self, x, z):

        x = self.bn(x)
        beta = self.lin_beta(z).view(len(x), self.planes, 1, 1)
        gamma = self.lin_gamma(z).view(len(x), self.planes, 1, 1)

        return (gamma + 1.) * x + beta


class ConditionalResUp(nn.Module):

    def __init__(self, planes, dilution=1, expansion=1, upsample=1):
        super(ConditionalResUp, self).__init__()

        self.conditional_dim = args.conditional_dim

        self.bn1 = ConditionalBatchNorm(planes, self.conditional_dim,
                                       track_running_stats=True, momentum=0.01)

        self.conv1 = nn.Sequential(
            nn.ReLU(),
            UpSample(upsample),
            spectral_norm(nn.Conv2d(planes, planes // dilution * expansion, kernel_size=3,
                                    padding=1, bias=False)),
            )

        self.bn2 = ConditionalBatchNorm(planes // dilution * expansion, self.conditional_dim,
                                        track_running_stats=True, momentum=0.01)

        self.conv2 = nn.Sequential(
            nn.ReLU(),
            spectral_norm(nn.Conv2d(planes // dilution * expansion, planes // dilution * expansion,
                                    kernel_size=3, padding=1, bias=False)),

        )

        if dilution * expansion == 1:
            self.identity = UpSample(upsample)
        elif expansion == 1:
            self.identity = nn.Sequential(UpSample(upsample),
                                          spectral_norm(nn.Conv2d(planes, planes // dilution, kernel_size=1, bias=False)))
        else:
            self.identity = nn.Sequential(UpSample(upsample),
                                          SelfExpansion(planes, expansion),)

    def forward(self, xz):

        x, z = xz
        zc, z = z[:, :self.conditional_dim], z[:, self.conditional_dim:]

        r = self.bn1(x, zc)
        r = self.conv1(r)
        r = self.bn2(r, zc)
        r = self.conv2(r)

        x = self.identity(x)

        return x + r, z


class Attention(nn.Module):

    def __init__(self, planes):
        super(Attention, self).__init__()

        self.query = nn.Sequential(
            nn.BatchNorm1d(planes, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            spectral_norm(nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False)),
        )

        self.key = nn.Sequential(
            nn.BatchNorm1d(planes, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            spectral_norm(nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False)),
        )

        self.value = nn.Sequential(
            nn.BatchNorm1d(planes, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            spectral_norm(nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False)),
        )

        self.output = nn.Sequential(
            nn.BatchNorm2d(planes, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(planes, planes, kernel_size=1, padding=0, bias=False)),
        )

        self.planes = planes

    def forward(self, x):

        b, c, h, w = x.shape

        xv = x.view(b, c, h*w)

        q = self.query(xv).transpose(1, 2)
        k = self.key(xv)
        v = self.value(xv).transpose(1, 2)

        a = torch.softmax(torch.bmm(q, k) / math.sqrt(self.planes), dim=2)
        r = torch.bmm(a, v).transpose(1, 2).view(b, c, h, w)
        r = self.output(r)

        return x + r


class AttributeHead(nn.Module):

    def __init__(self, planes):
        super(AttributeHead, self).__init__()

        # self.attention = nn.Sequential(
        #     nn.BatchNorm1d(planes, affine=True, track_running_stats=True, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Conv1d(planes, planes, kernel_size=1, padding=0, bias=False),
        # )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(planes, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(planes, planes // 2, bias=False),
            nn.BatchNorm1d(planes // 2, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(planes // 2, 1, bias=True),
        )

    def forward(self, x):

        # b, c, h, w = x.shape
        #
        # x = x.view(b, c, h*w)
        #
        # a = torch.softmax(self.attention(x), dim=2)
        # x = (x * a).sum(dim=2)
        y = self.fc(x)
        # y = self.fc2(y)

        return y


class ResUp(nn.Module):

    def __init__(self, planes, dilution=1, expansion=1, upsample=1):
        super(ResUp, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(planes, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            UpSample(upsample),
            spectral_norm(nn.Conv2d(planes, planes // dilution * expansion, kernel_size=3, padding=1, bias=False)),
            nn.BatchNorm2d(planes // dilution * expansion, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(planes // dilution * expansion, planes // dilution * expansion, kernel_size=3, padding=1, bias=False)),
            # nn.Dropout(0.1),
        )

        if dilution * expansion == 1:
            self.identity = UpSample(upsample)
        elif expansion == 1:
            self.identity = nn.Sequential(UpSample(upsample),
                                          spectral_norm(nn.Conv2d(planes, planes // dilution, kernel_size=1, bias=True)))
        else:
            self.identity = nn.Sequential(UpSample(upsample),
                                          SelfExpansion(planes, expansion),)

    def forward(self, x):

        h = self.conv(x)
        x = self.identity(x)

        return x + h


class ResClassifier(nn.Module):
    """DCGAN Discriminator D(z)"""

    def __init__(self, n=1):
        super(ResClassifier, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, args.channel, kernel_size=3, stride=1, padding=1, bias=True),
            ResNormDown(args.channel, expansion=2, downsample=2),
            ResNormDown(2 * args.channel, expansion=2, downsample=2),
            ResNormDown(4 * args.channel, expansion=2, downsample=2),
            ResNormDown(8 * args.channel, expansion=1, downsample=2),
            ResNormDown(8 * args.channel, expansion=2, downsample=2),
            ResNormDown(16 * args.channel, expansion=1, downsample=2),
            ResNormDown(16 * args.channel, expansion=1, downsample=1),
            nn.BatchNorm2d(16 * args.channel, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(16 * args.channel, 32 * args.channel, kernel_size=(4, 3), padding=0, bias=False),
            nn.Flatten()
            # nn.MaxPool2d((4, 3)),
            # nn.AvgPool2d((4, 3)),
            # nn.Flatten()
        )

        self.n = n
        # self.output = nn.Linear(16 * args.channel, n, bias=True)

        self.classifiers = nn.ModuleList([AttributeHead(32 * args.channel) for _ in range(self.n)])

        self.padding = (7, 7, 19, 19)
        init_weights(self)

    def freeze(self, fr=True):
        for param in self.parameters():
            param.requires_grad = not fr

    def forward(self, x):

        x = F.pad(x, self.padding)
        x = self.conv(x)

        # y = self.output(x)
        y = torch.cat([head(x) for head in self.classifiers], dim=1)

        if self.n == 1:
            y = y.view(-1)
        return y


class ResDiscriminator(nn.Module):
    """DCGAN Discriminator D(z)"""

    def __init__(self, n=1):
        super(ResDiscriminator, self).__init__()

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(3, args.channel, kernel_size=3, stride=1, padding=1, bias=True)),
            ResDown(args.channel, expansion=2, downsample=2),
            ResDown(2 * args.channel, expansion=2, downsample=2),
            # Attention(4 * args.channel),
            ResDown(4 * args.channel, expansion=2, downsample=2),
            ResDown(8 * args.channel, expansion=1, downsample=2),
            # Attention(8 * args.channel),
            ResDown(8 * args.channel, expansion=2, downsample=2),
            ResDown(16 * args.channel, expansion=1, downsample=2),
            ResDown(16 * args.channel, expansion=1, downsample=1),
            nn.ReLU(),
            nn.AvgPool2d((4, 3)),
            nn.Flatten()
        )

        self.n = n
        self.output = spectral_norm(nn.Linear(16 * args.channel, n, bias=False))

        self.padding = (7, 7, 19, 19)
        init_weights(self)

    def freeze(self, fr=True):
        for param in self.parameters():
            param.requires_grad = not fr

    def forward(self, x):

        x = F.pad(x, self.padding)
        x = self.conv(x)
        y = self.output(x)

        if self.n == 1:
            y = y.view(-1)
        return y


class ResGenerator(nn.Module):
    """DCGAN Generator G(z)"""

    def __init__(self):
        super(ResGenerator, self).__init__()

        self.latent_dim = args.latent_dim
        # self.latent_dim = args.latent_dim - 6 * args.conditional_dim

        # Project and reshape
        self.dconv = nn.Sequential(
            GPAttrLayer('view', -1, self.latent_dim, 1, 1),
            spectral_norm(nn.ConvTranspose2d(self.latent_dim, 16 * args.channel, kernel_size=(4, 3), bias=False)),
        )

        # self.res1 = nn.Sequential(
        #
        #     ConditionalResUp(16 * args.channel, dilution=1, upsample=2),
        #     ConditionalResUp(16 * args.channel, dilution=2, upsample=2),
        #     ConditionalResUp(8 * args.channel, dilution=1, upsample=2),
        # )
        # self.attention = Attention(8 * args.channel)
        # self.res2 = nn.Sequential(
        #     ConditionalResUp(8 * args.channel, dilution=2, upsample=2),
        #     ConditionalResUp(4 * args.channel, dilution=2, upsample=2),
        #     ConditionalResUp(2 * args.channel, dilution=2, upsample=2),
        #
        # )

        self.res = nn.Sequential(
            ResUp(16 * args.channel, dilution=1, upsample=2),
            ResUp(16 * args.channel, dilution=2, upsample=2),
            ResUp(8 * args.channel, dilution=1, upsample=2),
            # Attention(8 * args.channel),
            ResUp(8 * args.channel, dilution=2, upsample=2),
            # Attention(4 * args.channel),
            ResUp(4 * args.channel, dilution=2, upsample=2),
            ResUp(2 * args.channel, dilution=2, upsample=2),
        )

        self.output = nn.Sequential(
            nn.BatchNorm2d(args.channel, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(args.channel, 3, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.Tanh()
            )

        self.width = args.width
        self.height = args.height
        init_weights(self)

    def freeze(self, fr=True):
        for param in self.parameters():
            param.requires_grad = fr

    def forward(self, z):

        # z, zc = z[:, :self.latent_dim], z[:, self.latent_dim:]

        x = self.dconv(z)

        x = self.res(x)
        # x, zc = self.res1((x, zc))
        # x = self.attention(x)
        # x, _ = self.res2((x, zc))

        x = self.output(x)

        x = x.narrow(2, 19, self.height).narrow(3, 7, self.width)
        return x


class Generator(nn.Module):
    """DCGAN Generator G(z)"""

    def __init__(self):
        super(Generator, self).__init__()

        # Project and reshape
        self.dconv = nn.Sequential(

            GPAttrLayer('view', -1, args.latent_dim, 1, 1),
            nn.ConvTranspose2d(args.latent_dim, 512, kernel_size=(4, 3), bias=False),

            nn.InstanceNorm2d(512, affine=False, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=False, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=False, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            )

        self.width = args.width
        self.height = args.height
        self.init_weights()

    def init_weights(self):

        n = len(list(self.dconv.parameters()))

        for i, p in enumerate(self.dconv.parameters()):
            if i == n-1:
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='tanh')
            else:
                nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')

    def freeze(self, fr=True):
        for param in self.parameters():
            param.requires_grad = fr

    def forward(self, x):
        x = self.dconv(x)
        x = x.narrow(2, 19, self.height).narrow(3, 7, self.width)
        return x


class Discriminator(nn.Module):
    """DCGAN Discriminator D(z)"""

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True, track_running_stats=True, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=(4, 3), stride=1, padding=0, bias=False),

        )

        self.padding = (7, 7, 19, 19)

    def init_weights(self):

        for i, p in enumerate(self.modules()):

            if isinstance(p, nn.Conv2d):

                if p.weight.shape[0] == 1:
                    nn.init.kaiming_normal_(p.weight, mode='fan_out', nonlinearity='linear')
                else:
                    nn.init.kaiming_normal_(p.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):

        x = F.pad(x, self.padding)
        y = self.conv(x)
        return y.view(-1)


class MiniGenerator(nn.Module):
    """DCGAN Generator G(z)"""

    def __init__(self):
        super(MiniGenerator, self).__init__()

        # Project and reshape
        self.linear = nn.Sequential(
            nn.Linear(args.latent_dim, 512 * 4 * 4, bias=False),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(inplace=True))

        # Upsample
        self.features = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            )

    def forward(self, x):
        x = self.linear(x).view(x.size(0), -1, 4, 4)
        return self.features(x)


class MiniDiscriminator(nn.Module):
    """DCGAN Discriminator D(z)"""

    def __init__(self):
        super(MiniDiscriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, bias=False))

    def forward(self, x):
        return self.features(x).view(-1)

    def clip(self, c=0.05):
        """Weight clipping in (-c, c)"""

        for p in self.parameters():
            p.data.clamp_(-c, c)

def init_weights(net):
    net.param_count = 0
    for module in net.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.ConvTranspose2d, nn.ConvTranspose1d)):
            if args.init == 'ortho':
                torch.nn.init.orthogonal_(module.weight)
            elif args.init == 'N02':
                torch.nn.init.normal_(module.weight, 0, 0.02)
            elif args.init in ['glorot', 'xavier']:
                torch.nn.init.xavier_uniform_(module.weight)
            else:
                print('Init style not recognized...')
        net.param_count += sum([p.data.nelement() for p in module.parameters()])
