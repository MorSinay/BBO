from config import args, exp
import torch
from torch import nn
import torch.nn.functional as F
import copy
from collections import defaultdict
from apex import amp
from loguru import logger


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
        self.delta = args.delta
        self.batch = args.batch
        self.replay_memory_size = self.batch * args.replay_memory_factor
        self.problem_index = args.problem_index
        self.pi_lr = args.pi_lr
        self.value_lr = args.value_lr
        self.grad_steps = args.grad_steps
        self.importance_sampling = args.importance_sampling
        self.update_step = args.update_step
        self.best_explore_update = args.best_explore_update
        self.weight_decay = args.weight_decay
        self.warm_up = args.warm_up
        self.grad = args.grad
        self.cone_angle = args.cone_angle
        self.warmup_factor = args.warmup_factor

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
        try:
            state['amp'] = amp.state_dict()
        except:
            pass

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

        try:
            amp.load_state_dict(state['amp'])
        except Exception as e:
            logger.error(str(e))

        return state['aux']



