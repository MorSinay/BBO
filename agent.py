import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
import torch.nn as nn
from collections import defaultdict
from torchvision.utils import save_image

from config import consts, args, DirsAndLocksSingleton
from model_ddpg import DuelNet, DerivativeNet, PiNet

import os
import copy
mem_threshold = consts.mem_threshold


class Agent(object):

    def __init__(self, exp_name, env, checkpoint):

        reward_str = "BBO"
        print("Learning POLICY method using {} with BBOAgent".format(reward_str))

        self.env = env
        self.dirs_locks = DirsAndLocksSingleton(exp_name)
        self.action_space = args.action_space
        self.epsilon = args.epsilon
        self.delta = args.delta
        self.cuda_id = args.cuda_default
        self.batch = args.batch
        self.warmup_minibatch = 4
        self.replay_memory_size = self.batch*args.replay_memory_factor
        self.replay_memory_factor = args.replay_memory_factor
        self.problem_index = args.problem_index
        self.value_lr = args.value_lr
        self.budget = args.budget
        self.checkpoint = checkpoint
        self.algorithm_method = args.algorithm
        self.grad_steps = args.grad_steps
        self.stop_con = args.stop_con
        self.grad_clip = args.grad_clip
        self.divergence = 0
        self.importance_sampling = args.importance_sampling
        self.bandage = args.bandage
        self.update_step = args.update_step
        self.best_explore_update = args.best_explore_update
        self.analysis_dir = os.path.join(self.dirs_locks.analysis_dir, str(self.problem_index))
        if not os.path.exists(self.analysis_dir):
            try:
                os.makedirs(self.analysis_dir)
            except:
                pass

        self.frame = 0
        self.n_offset = 0
        self.results = defaultdict(list)
        self.clip = args.clip
        self.tensor_replay_reward = None
        self.tensor_replay_policy = None
        self.mean = None
        self.std = None
        self.lr_list = [1e-2, 1e-3, 1e-4]
        self.lr_index = 0
        self.pi_lr = self.lr_list[self.lr_index]

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        if args.explore == 'grad_rand':
            self.exploration = self.exploration_grad_rand
        elif args.explore == 'grad_direct':
            self.exploration = self.exploration_grad_direct
        elif args.explore == 'grad_prop':
            self.exploration = self.exploration_grad_direct_prop
        elif args.explore == 'rand':
            self.exploration = self.exploration_rand
        else:
            print("explore:"+args.explore)
            raise NotImplementedError

        self.init = torch.tensor(self.env.get_initial_solution(), dtype=torch.float).to(self.device)
        self.pi_net = PiNet(self.init, self.device, self.action_space)
        self.optimizer_pi = torch.optim.SGD([self.pi_net.pi], lr=self.lr_list[self.lr_index])

        if self.algorithm_method in ['first_order', 'second_order']:
            self.value_iter = 100
            self.derivative_net = DerivativeNet()
            self.derivative_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_derivative = torch.optim.Adam(self.derivative_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
        elif self.algorithm_method == 'value':
            self.value_iter = 40
            self.value_net = DuelNet()
            self.value_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
        elif self.algorithm_method == 'anchor':
            self.value_iter = 100
            self.derivative_net = DerivativeNet()
            self.derivative_net.to(self.device)
            self.value_net = DuelNet()
            self.value_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_derivative = torch.optim.Adam(self.derivative_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
        else:
            raise NotImplementedError

        self.q_loss = nn.SmoothL1Loss(reduction='none')

    def update_pi_optimizer_lr(self):
        op_dict = self.optimizer_pi.state_dict()
        #self.lr_index = (self.lr_index + 1)%len(self.lr_list)
        self.lr_index = min((self.lr_index + 1) , len(self.lr_list)-1)
        self.pi_lr = self.lr_list[self.lr_index]
        op_dict['param_groups'][0]['lr'] = self.pi_lr
        self.optimizer_pi.load_state_dict(op_dict)

    def save_results(self):
        for k in self.results.keys():
            path = os.path.join(self.analysis_dir, k +'.npy')
            if k in ['explore_policies']:
                policy = self.pi_net(torch.cat(self.results[k], dim=0))
                assert (len(policy.shape) == 2), "save_results"
                np.save(path, policy.cpu().numpy())
            elif k in ['policies']:
                policy = self.pi_net(torch.stack(self.results[k]))
                assert (len(policy.shape) == 2), "save_results"
                np.save(path, policy.cpu().numpy())
            else:
                tmp = np.array(self.results[k]).flatten()
                if tmp is None:
                    assert False, "save_results"
                np.save(path, tmp)

        path = os.path.join(self.analysis_dir, 'f0.npy')
        np.save(path, self.env.get_f0())

        if self.action_space == 784:
            path = os.path.join(self.analysis_dir, 'reconstruction.png')
            save_image(self.pi_net.pi.cpu().view(1, 28, 28), path)

    def save_checkpoint(self, path, aux=None):
        if self.algorithm_method in ['first_order', 'second_order']:
            state = {'pi_net': self.pi_net,
                     'derivative_net': self.derivative_net.state_dict(),
                     'optimizer_derivative': self.optimizer_derivative.state_dict(),
                     'optimizer_pi': self.optimizer_pi.state_dict(),
                     'aux': aux}
        elif self.algorithm_method == 'value':
            state = {'pi_net': self.pi_net,
                     'value_net': self.value_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_pi': self.optimizer_pi.state_dict(),
                     'aux': aux}
        elif self.algorithm_method == 'anchor':
            state = {'pi_net': self.pi_net,
                     'derivative_net': self.derivative_net.state_dict(),
                     'optimizer_derivative': self.optimizer_derivative.state_dict(),
                     'value_net': self.value_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_pi': self.optimizer_pi.state_dict(),
                     'aux': aux}
        else:
            raise NotImplementedError

        torch.save(state, path)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            assert False, "load_checkpoint"
        state = torch.load(path, map_location="cuda:%d" % self.cuda_id)
        self.pi_net = state['pi_net'].to(self.device)
        self.optimizer_pi.load_state_dict(state['optimizer_pi'])
        if self.algorithm_method in ['first_order', 'second_order']:
            self.derivative_net.load_state_dict(state['derivative_net'])
            self.optimizer_derivative.load_state_dict(state['optimizer_derivative'])
        elif self.algorithm_method == 'value':
            self.value_net.load_state_dict(state['value_net'])
            self.optimizer_value.load_state_dict(state['optimizer_value'])
        elif self.algorithm_method == 'anchor':
            self.derivative_net.load_state_dict(state['derivative_net'])
            self.optimizer_derivative.load_state_dict(state['optimizer_derivative'])
            self.value_net.load_state_dict(state['value_net'])
            self.optimizer_value.load_state_dict(state['optimizer_value'])
        else:
            raise NotImplementedError
        self.n_offset = state['aux']['n']

        return state['aux']

    def update_mean_std(self, mean, std):
        self.mean = mean
        self.std = std
        #self.mean = 0
        #self.std = 1
        self.value_optimize()

    def exploration_rand(self, n_explore):
            pi = self.pi_net.pi.detach().clone().cpu()
            pi_explore = pi + self.epsilon * torch.randn(n_explore, self.action_space)
            return pi_explore

    def exploration_grad_rand(self, n_explore):
        grads = self.get_grad().cpu()
        pi = self.pi_net.pi.detach().clone().cpu()
        explore_factor = self.delta * grads + self.epsilon * torch.randn(n_explore, self.action_space)
        explore_factor *= 0.9 ** (2 * torch.arange(n_explore, dtype=torch.float)).reshape(n_explore, 1)
        pi_explore = pi + explore_factor  # gradient decent
        return pi_explore

    def exploration_grad_direct(self, n_explore):
        pi_array = self.get_n_grad_ahead(self.grad_steps).reshape(self.grad_steps, self.action_space).cpu()
        n_explore = (n_explore // self.grad_steps)

        epsilon_array = 0.01 ** (3 - 2 * torch.arange(self.grad_steps, dtype=torch.float) / (self.grad_steps - 1))
        epsilon_array = epsilon_array.unsqueeze(1) # .expand_dims(epsilon_array, axis=1)
        pi_explore = torch.cat([pi_array + epsilon_array * torch.randn(self.grad_steps, self.action_space) for _ in range(n_explore)], dim=0)
        return pi_explore

    def exploration_grad_direct_prop(self, n_explore):
        pi_array = self.get_n_grad_ahead(self.grad_steps + 1).cpu()
        n_explore = (n_explore // self.grad_steps)

        diff_pi = torch.norm(pi_array[1:] - pi_array[:-1], 2, dim=1)

        epsilon_array = diff_pi * 10 ** (torch.arange(self.grad_steps, dtype=torch.float) / (self.grad_steps - 1))
        epsilon_array = epsilon_array.unsqueeze(1)#np.expand_dims(epsilon_array, axis=1)
        pi_explore = torch.cat([pi_array[:-1] + epsilon_array * torch.randn(self.grad_steps, self.action_space) for _ in range(n_explore)], dim=0)

        return pi_explore

    def exploration_step(self, n_explore):
        pi_explore = self.exploration(n_explore)
        self.step_policy(self.pi_net(pi_explore).detach())
        rewards = self.env.reward
        if self.best_explore_update:
            best_explore = rewards.argmax()
            self.pi_net.pi_update(pi_explore[best_explore].to(self.device))

        return pi_explore, rewards

    def get_grad(self):
        self.pi_net.eval()
        if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
            self.derivative_net.eval()
            grad = self.derivative_net(self.pi_net()).detach().squeeze(0)
            return grad
        elif self.algorithm_method == 'value':
            self.value_net.eval()
            self.optimizer_pi.zero_grad()
            loss_pi = self.value_net(self.pi_net())
            loss_pi.backward()
            return self.pi_net.pi.grad.detach()
        else:
            raise NotImplementedError

    def f_policy(self, policy):
        policy = policy.data.cpu().numpy()
        assert (policy.max() <= 1), "policy.max() {}".format(policy.max())
        assert (policy.min() >= -1), "policy.min() {}".format(policy.min())
        return self.env.f(policy)

    def step_policy(self, policy):
        policy = policy.data.cpu().numpy()
        assert (policy.max() <= 1), "policy.max() {}".format(policy.max())
        assert (policy.min() >= -1), "policy.min() {}".format(policy.min())
        self.env.step_policy(policy)

    def pi_optimize(self):
        pi = self.pi_net.pi.detach().clone()
        pi_list = [pi]
        value_list = [self.f_policy(pi)]

        for _ in range(self.grad_steps):
            pi, value = self.grad_step()
            pi_list.append(pi)
            value_list.append(value)

        if self.update_step == 'n_step':
            #no need to do action
            pass
        elif self.update_step == 'best_step':
            best_idx = np.array(value_list).argmin()
            self.pi_net.pi_update(pi_list[best_idx].to(self.device))
        elif self.update_step == 'first_vs_last':
            if value_list[-1] > value_list[0]:
                self.pi_net.pi_update(pi_list[0].to(self.device))
        elif self.update_step == 'no_update':
            self.pi_net.pi_update(pi_list[0].to(self.device))
        else:
            raise NotImplementedError

    def grad_step(self):
        self.pi_net.train()
        self.optimizer_pi.zero_grad()
        if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
            self.derivative_net.eval()
            grad = self.derivative_net(self.pi_net()).detach().squeeze(0)
            self.pi_net.grad_update(-grad.clone())
        elif self.algorithm_method == 'value':
            self.value_net.eval()
            loss_pi = self.value_net(self.pi_net())
            loss_pi.backward()
        else:
            raise NotImplementedError

        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.pi_net.pi, 1e-2 / self.pi_lr)

        self.optimizer_pi.step()
        self.pi_net.eval()
        pi = self.pi_net.pi.detach().clone()
        return pi, self.f_policy(pi)

    def get_n_grad_ahead(self, n):

        optimizer_state = copy.deepcopy(self.optimizer_pi.state_dict())
        pi_array = [self.pi_net.pi.detach().clone()]
        for _ in range(n-1):
            pi, _ = self.grad_step()
            pi_array.append(pi)

        self.optimizer_pi.load_state_dict(optimizer_state)
        self.pi_net.pi_update(pi_array[0].to(self.device))

        pi_array = torch.stack(pi_array)
        return pi_array

    def warmup(self):

        explore_policies = self.exploration_rand(2 * self.batch)
        self.step_policy(self.pi_net(explore_policies).detach())
        rewards = self.env.reward
        self.results['explore_policies'].append(explore_policies)
        self.results['rewards'].append(rewards)
        self.update_mean_std(rewards.mean(), rewards.std())

    def norm(self, x):
        x = (x - self.mean) / (self.std + 1e-5)
        return x

    def denorm(self, x):
        x = (x * (self.std + 1e-5)) + self.mean
        return x

    def update_best_pi(self):
        rewards = np.hstack(self.results['rewards'])
        best_idx = rewards.argmax()
        pi = torch.cat(self.results['explore_policies'], dim=0)[best_idx]
        self.pi_net.pi_update(pi.to(self.device))

    def value_optimize(self):
        raise NotImplementedError

    def minimize(self, n_explore):
        raise NotImplementedError