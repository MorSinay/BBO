from config import args, exp
import torch
from torch import nn
import torch.nn.functional as F
import copy
from collections import defaultdict
# from apex import amp
from loguru import logger
from environment2 import Env
import math
from model import DuelNet, DuelResNet, PiNet, SplineNet, MultipleOptimizer

if args.architecture == 'fc':
    NeuralNet = DuelNet
elif args.architecture == 'res':
    NeuralNet = DuelResNet
elif args.architecture == 'spline':
    NeuralNet = SplineNet
else:
    raise NotImplementedError


class Algorithm(object):

    def __init__(self):
        self.networks_dict = {}
        self.optimizers_dict = {}

        self.half = args.half
        self.device = exp.device
        self.train_epoch = args.train_epoch
        self.test_epoch = args.test_epoch
        self.multi_gpu = args.multi_gpu

        self.batch = args.batch
        self.clip = args.clip

        self.action_space = args.action_space
        self.epsilon = args.epsilon
        self.batch = args.batch
        self.n_explore = args.n_explore
        self.replay_memory_size = self.n_explore * args.replay_memory_factor
        self.problem_index = args.problem_index
        self.pi_lr = args.pi_lr
        self.value_lr = args.value_lr
        self.importance_sampling = args.importance_sampling
        self.best_explore_update = args.best_explore_update
        self.weight_decay = args.weight_decay
        self.cone_angle = args.cone_angle
        self.warmup_factor = args.warmup_factor
        self.algorithm = args.algorithm
        self.grad_clip = args.grad_clip
        self.stop_con = args.stop_con
        self.epsilon_factor = args.epsilon_factor
        self.warmup_minibatch = args.warmup_minibatch
        self.budget = args.budget

        if args.explore == 'rand':
            self.exploration = self.exploration_rand
        elif args.explore == 'ball':
            self.exploration = self.ball_explore
        elif args.explore == 'cone':
            self.exploration = self.cone_explore
        elif args.explore == 'cone_rand':
            self.exploration = self.cone_rand_explore
        else:
            print("explore:" + args.explore)
            raise NotImplementedError

        self.env = Env(self.problem_index)
        self.pi_0 = self.env.get_initial_solution()

        self.pi_net = PiNet(self.pi_0, self.device, self.action_space)
        self.optimizer_pi = torch.optim.SGD([self.pi_net.pi], lr=self.pi_lr)
        self.pi_net.eval()
        
        self.value_iter = args.learn_iteration

        if self.algorithm == 'egl':
            output = self.action_space
        elif self.algorithm == 'igl':
            output = 1
        else:
            raise NotImplementedError

        self.net = NeuralNet(self.pi_net, output=output)
        self.net.to(self.device)

        if args.architecture == 'spline':

            opt_sparse = torch.optim.SparseAdam(self.net.embedding.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-04)
            opt_dense = torch.optim.Adam(self.net.head.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-04)
            self.optimizer = MultipleOptimizer(opt_sparse, opt_dense)

        else:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)

        self.net_zero = copy.deepcopy(self.net.state_dict())

        if args.loss == 'huber':
            self.q_loss = nn.SmoothL1Loss(reduction='none')
        elif args.loss == 'mse':
            self.q_loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError

    def postprocess(self, sample):

        for name, var in sample.items():
            sample[name] = var.to(self.device)

        return sample

    def reset_opt(self, optimizer):

        optimizer.state = defaultdict(dict)

    def reset_networks(self, networks_dict, optimizers_dict):

        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        for net in networks_dict:
            net = getattr(self, net)
            net.apply(init_weights)

        for optim in optimizers_dict:
            optim = getattr(self, optim)
            optim.state = defaultdict(dict)

    def get_optimizers(self):

        self.optimizers_dict = {}

        for d in dir(self):
            x = getattr(self, d)
            if issubclass(type(x), torch.optim.Optimizer) and hasattr(x, 'state_dict'):
                self.optimizers_dict[d] = x

        return self.optimizers_dict

    def get_networks(self):

        self.networks_dict = {}
        name_dict = {}

        for d in dir(self):
            x = getattr(self, d)
            if issubclass(type(x), nn.Module) and hasattr(x, 'state_dict'):
                if next(x.parameters(), None) is not None:
                    name_dict[d] = getattr(x, 'named_parameters')
                    self.networks_dict[d] = x

        return name_dict

    def train(self):

        if not self.networks_dict:
            self.get_networks()

        for net in self.networks_dict.values():
            net.train()

    def eval(self):

        if not self.networks_dict:
            self.get_networks()

        for net in self.networks_dict.values():
            net.eval()

    def state_dict(self, net):

        if self.multi_gpu:
            return copy.deepcopy(net.module.state_dict())
        return copy.deepcopy(net.state_dict())

    def load_state_dict(self, net, state):

        if self.multi_gpu:
            net.module.load_state_dict(state, strict=False)
        else:
            net.load_state_dict(state, strict=False)

    def store_net_0(self):

        if not self.networks_dict:
            self.get_networks()

        self.net_0 = {}

        for name, net in self.networks_dict.items():
            net.eval()

            self.net_0[name] = self.state_dict(net)

    def save_checkpoint(self, path=None, aux=None):

        if not self.networks_dict:
            self.get_networks()
        if not self.optimizers_dict:
            self.get_optimizers()

        state = {'aux': aux}

        for net in self.networks_dict:
            state[net] = self.state_dict(self.networks_dict[net])

        for optimizer in self.optimizers_dict:
            state[optimizer] = copy.deepcopy(self.optimizers_dict[optimizer].state_dict())

        if path is not None:
            torch.save(state, path)

        return state

    def load_checkpoint(self, pathstate):

        if not self.networks_dict:
            self.get_networks()
            self.get_optimizers()

        if type(pathstate) is str:
            state = torch.load(pathstate, map_location=self.device)
        else:
            state = pathstate

        for net in self.networks_dict:
            self.load_state_dict(self.networks_dict[net], state[net])

        for optimizer in self.optimizers_dict:
            self.optimizers_dict[optimizer].load_state_dict(state[optimizer])
            # pass

        return state['aux']

    def reset_net(self):
        self.net.load_state_dict(self.net_zero)
        self.optimizer.state = defaultdict(dict)

    def exploration_rand(self, n_explore):
        pi = self.pi_net.pi.detach().clone()
        rand_sign = (2*torch.randint(0, 2,size=(n_explore-1, self.action_space), device=self.device)-1).reshape(n_explore-1, self.action_space)
        pi_explore = pi + self.warmup_factor*self.epsilon * rand_sign * torch.cuda.FloatTensor(n_explore-1, self.action_space).uniform_()
        return torch.cat([pi.unsqueeze(0), pi_explore], dim=0)

    def cone_explore_(self, n_explore, angle, pi, grad):
        alpha = math.pi/angle
        pi = pi.unsqueeze(0)

        x = torch.cuda.FloatTensor(n_explore, self.action_space).normal_()
        mag = torch.cuda.FloatTensor(n_explore, 1).uniform_()

        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        grad = grad / (torch.norm(grad) + 1e-8)

        cos = (x @ grad).unsqueeze(1)

        dp = x - cos * grad.unsqueeze(0)

        dp = dp / torch.norm(dp, dim=1, keepdim=True)

        acos = torch.acos(torch.clamp(torch.abs(cos), 0, 1-1e-8))

        new_cos = torch.cos(acos * alpha / (math.pi / 2))
        new_sin = torch.sin(acos * alpha / (math.pi / 2))

        cone = new_sin * dp + new_cos * grad

        explore = pi - self.epsilon * math.sqrt(self.action_space) * mag * cone

        return explore

    def cone_rand_explore(self, n_explore):
        pi, grad = self.get_grad(grad_step=False)

        # explore_rand = self.ball_explore_(n_explore//2, 1, pi, grad)
        explore_rand = self.ball_explore_(pi, n_explore // 2)
        explore_cone = self.cone_explore_(n_explore - n_explore // 2 - 1, self.cone_angle, pi, grad)

        return torch.cat([pi.unsqueeze(0), explore_rand, explore_cone], dim=0)

    def cone_explore(self, n_explore):
        pi, grad = self.get_grad(grad_step=False)

        explore_cone = self.cone_explore_(n_explore - 1, self.cone_angle, pi, grad)

        return torch.cat([pi.unsqueeze(0), explore_cone], dim=0)

    def ball_explore_(self, pi, n_explore):
        pi = pi.unsqueeze(0)

        x = torch.cuda.FloatTensor(n_explore, self.action_space).normal_()
        mag = torch.cuda.FloatTensor(n_explore, 1).uniform_()

        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)

        explore = pi + self.epsilon * math.sqrt(self.action_space) * mag * x

        return explore

    def ball_explore(self, n_explore):

        pi = self.pi_net.pi.detach().clone()
        explore = self.ball_explore_(pi, n_explore - 1)

        return torch.cat([pi.unsqueeze(0), explore], dim=0)

    def get_grad(self, grad_step=False):

        self.pi_net.train()
        self.optimizer_pi.zero_grad()
        if self.algorithm == 'egl':
            self.optimizer.zero_grad()
            grad = self.net(self.pi_net.pi).view_as(self.pi_net.pi).detach().clone()

            self.pi_net.grad_update(grad)
        elif self.algorithm == 'igl':
            self.optimizer.zero_grad()
            loss_pi = self.net(self.pi_net.pi)
            loss_pi.backward()
        else:
            raise NotImplementedError

        if self.grad_clip != 0:
            nn.utils.clip_grad_norm_(self.pi_net.pi, self.grad_clip / self.pi_lr)

        if grad_step:
            self.optimizer_pi.step()

        pi = self.pi_net.pi.detach().clone()
        grad = self.pi_net.grad.detach().clone()
        return pi, grad

