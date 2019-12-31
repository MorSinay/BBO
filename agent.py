import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from collections import defaultdict
import torch.autograd as autograd
from torchvision.utils import save_image
from sklearn.preprocessing import RobustScaler
from config import consts, args, DirsAndLocksSingleton
from model_ddpg import DuelNet, DerivativeNet, PiNet, SplineNet, MultipleOptimizer
import math
import os
import copy
from visualize_2d import get_best_solution
import itertools

class Agent(object):

    def __init__(self, exp_name, env, checkpoint):
        self.cuda_id = args.cuda_default
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.action_space = args.action_space
        self.env = env
        self.dirs_locks = DirsAndLocksSingleton(exp_name)

        self.best_op_x, self.best_op_f = get_best_solution(self.action_space, self.env.problem_iter)
        self.best_op_x = torch.FloatTensor(self.best_op_x).to(self.device)

        self.batch = args.batch
        self.n_explore = args.n_explore
        self.warmup_minibatch = args.warmup_minibatch
        self.replay_memory_size = self.batch * args.replay_memory_factor
        self.replay_memory_factor = args.replay_memory_factor
        self.problem_index = env.problem_iter
        self.value_lr = args.value_lr
        self.budget = args.budget
        self.checkpoint = checkpoint
        self.algorithm_method = args.algorithm
        self.grad_steps = args.grad_steps
        self.stop_con = args.stop_con
        self.grad_clip = args.grad_clip
        self.divergence = 0
        self.importance_sampling = args.importance_sampling
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
        self.tensor_replay_reward = None
        self.tensor_replay_policy = None
        self.pi_lr = args.pi_lr
        self.epsilon = args.epsilon
        self.delta = self.pi_lr
        self.warmup_factor = args.warmup_factor

        if args.explore == 'grad_rand':
            self.exploration = self.exploration_grad_rand
        elif args.explore == 'grad_direct':
            self.exploration = self.exploration_grad_direct
        elif args.explore == 'rand':
            self.exploration = self.exploration_rand
        elif args.explore == 'rand_test':
            self.exploration = self.exploration_rand_test
        elif args.explore == 'cone':
            if self.action_space == 1:
                self.exploration = self.exploration_grad_direct
            else:
                self.exploration = self.cone_explore
        else:
            print("explore:" + args.explore)
            raise NotImplementedError

        self.init = torch.FloatTensor(self.env.get_initial_solution()).to(self.device)
        self.pi_net = PiNet(self.init, self.device, self.action_space)
        self.optimizer_pi = torch.optim.SGD([self.pi_net.pi], lr=self.pi_lr)
        self.pi_net.eval()

        self.value_iter = args.learn_iteration
        if self.algorithm_method in ['first_order', 'second_order']:
            self.derivative_net = DerivativeNet(self.pi_net)
            self.derivative_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_derivative = torch.optim.Adam(self.derivative_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.derivative_net.eval()
            self.derivative_net_zero = copy.deepcopy(self.derivative_net.state_dict())
        elif self.algorithm_method == 'value':
            self.value_net = DuelNet(self.pi_net)
            self.value_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.value_net.eval()
            self.value_net_zero = copy.deepcopy(self.value_net.state_dict())
        elif self.algorithm_method == 'spline':
            self.spline_net = SplineNet(self.device, self.pi_net, output=1)
            self.spline_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            opt_sparse = torch.optim.SparseAdam(self.spline_net.embedding.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-04)
            opt_dense = torch.optim.Adam(self.spline_net.head.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-04)
            self.optimizer_spline = MultipleOptimizer(opt_sparse, opt_dense)
            self.spline_net.eval()
            self.spline_net_zero = copy.deepcopy(self.spline_net.state_dict())
        elif self.algorithm_method == 'anchor':
            self.derivative_net = DerivativeNet(self.pi_net)
            self.derivative_net.to(self.device)
            self.value_net = DuelNet(self.pi_net)
            self.value_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_derivative = torch.optim.Adam(self.derivative_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.value_net.eval()
            self.derivative_net.eval()
            self.value_net_zero = copy.deepcopy(self.value_net.state_dict())
            self.derivative_net_zero = copy.deepcopy(self.derivative_net.state_dict())
        else:
            raise NotImplementedError

        self.q_loss = nn.SmoothL1Loss(reduction='none')

    def update_pi_optimizer_lr(self):
        op_dict = self.optimizer_pi.state_dict()
        self.pi_lr *= 0.85
        op_dict['param_groups'][0]['lr'] = self.pi_lr
        self.optimizer_pi.load_state_dict(op_dict)

    def save_results(self, normalize_policy=False):
        for k in self.results.keys():
            path = os.path.join(self.analysis_dir, k +'.npy')
            if k in ['explore_policies']:
                policy = torch.cat(self.results[k], dim=0)
                if normalize_policy:
                    policy = self.pi_net(policy)
                assert ((len(policy.shape) == 2) and (policy.shape[1] == self.action_space)), "save_results"
                np.save(path, policy.cpu().numpy())
            elif k in ['policies']:
                policy = torch.stack(self.results[k])
                if normalize_policy:
                    policy = self.pi_net(policy)
                assert ((len(policy.shape) == 2) and (policy.shape[1] == self.action_space)), "save_results"
                np.save(path, policy.cpu().numpy())
            elif k in ['grad']:
                grad = np.vstack(self.results[k])
                assert ((len(grad.shape) == 2) and (grad.shape[1] == self.action_space)), "save_results"
                np.save(path, grad)
            else:
                tmp = np.array(self.results[k]).flatten()
                if tmp is None:
                    assert False, "save_results"
                np.save(path, tmp)

        best_list, observed_list, _ = self.env.get_observed_and_pi_list()
        np.save(os.path.join(self.analysis_dir, 'best_list_with_explore.npy'), np.array(best_list))
        np.save(os.path.join(self.analysis_dir, 'observed_list_with_explore.npy'), np.array(best_list))

        path = os.path.join(self.analysis_dir, 'f0.npy')
        np.save(path, self.env.get_f0())

        if self.action_space == 784:
            path = os.path.join(self.analysis_dir, 'reconstruction.png')
            save_image(self.pi_net.pi.cpu().view(1, 28, 28), path)

    def save_checkpoint(self, path, aux=None):
        if self.algorithm_method in ['first_order', 'second_order']:
            state = {'pi_net': self.pi_net.pi.detach(),
                     'derivative_net': self.derivative_net.state_dict(),
                     'optimizer_derivative': self.optimizer_derivative.state_dict(),
                     'optimizer_pi': self.optimizer_pi.state_dict(),
                     'aux': aux}
        elif self.algorithm_method == 'value':
            state = {'pi_net': self.pi_net.pi.detach(),
                     'value_net': self.value_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_pi': self.optimizer_pi.state_dict(),
                     'aux': aux}
        elif self.algorithm_method == 'anchor':
            state = {'pi_net': self.pi_net.pi.detach(),
                     'derivative_net': self.derivative_net.state_dict(),
                     'optimizer_derivative': self.optimizer_derivative.state_dict(),
                     'value_net': self.value_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_pi': self.optimizer_pi.state_dict(),
                     'aux': aux}
        elif self.algorithm_method == 'spline':
            state = {'pi_net': self.pi_net.pi.detach(),
                     'spline_net': self.spline_net.state_dict(),
                     'optimizer_spline': self.optimizer_spline.state_dict(),
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
        elif self.algorithm_method == 'spline':
            self.spline_net.load_state_dict(state['spline_net'])
            self.optimizer_spline.load_state_dict(state['optimizer_spline'])
        else:
            raise NotImplementedError
        self.n_offset = state['aux']['n']

        return state['aux']

    def reset_net(self):
        if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
            self.derivative_net.load_state_dict(self.derivative_net_zero)
            self.optimizer_derivative.state = defaultdict(dict)
        if self.algorithm_method in ['value', 'anchor']:
            self.value_net.load_state_dict(self.value_net_zero)
            self.optimizer_value.state = defaultdict(dict)
        if self.algorithm_method in ['spline']:
            self.spline_net.load_state_dict(self.spline_net_zero)
            self.optimizer_spline.state = defaultdict(dict)

    def get_n_grad_ahead(self, n):

        optimizer_state = copy.deepcopy(self.optimizer_pi.state_dict())
        pi_array = [self.pi_net.pi.detach().clone()]
        for _ in range(n):
            pi, _ = self.grad_step()
            pi_array.append(pi)

        self.optimizer_pi.load_state_dict(optimizer_state)
        self.pi_net.pi_update(pi_array[0].to(self.device))

        pi_array = torch.stack(pi_array)
        return pi_array

    def grad_step(self):
        self.pi_net.train()
        self.optimizer_pi.zero_grad()
        if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
            self.optimizer_derivative.zero_grad()
            grad = self.derivative_net(self.pi_net.pi).detach().squeeze(0)
            self.pi_net.grad_update(grad.clone())
        elif self.algorithm_method == 'value':
            self.optimizer_value.zero_grad()
            loss_pi = self.value_net(self.pi_net.pi)
            loss_pi.backward()
        elif self.algorithm_method == 'spline':
            self.optimizer_spline.zero_grad()
            loss_pi = self.spline_net(self.pi_net.pi)
            loss_pi.backward()
        else:
            raise NotImplementedError

        if self.grad_clip != 0:
            nn.utils.clip_grad_norm_(self.pi_net.pi, self.grad_clip / self.pi_lr)

        self.optimizer_pi.step()
        self.pi_net.eval()
        pi = self.pi_net.pi.detach().clone()
        return pi, self.step_policy(pi, to_env=False)

    def exploration_rand(self, n_explore):
            pi = self.pi_net.pi.detach().clone().cpu()
            pi_explore = pi + self.warmup_factor*self.epsilon * torch.rand(n_explore, self.action_space)
            return pi_explore

    def exploration_rand_test(self, n_explore):
            pi_explore = -2 * torch.rand(n_explore, self.action_space) + 1
            return pi_explore

    def exploration_grad_rand(self, n_explore):
        grads = self.get_grad().cpu()
        pi = self.pi_net.pi.detach().clone().cpu()
        explore_factor = self.delta * grads + self.epsilon * torch.randn(n_explore, self.action_space)
        explore_factor *= 0.9 ** (2 * torch.arange(n_explore, dtype=torch.float)).reshape(n_explore, 1)
        pi_explore = pi - explore_factor  # gradient decent
        return pi_explore

    def exploration_grad_direct(self, n_explore):
        pi_array = self.get_n_grad_ahead(self.grad_steps).reshape(self.grad_steps+1, self.action_space).cpu()
        n_explore = (n_explore // self.grad_steps)

        epsilon_array = self.epsilon ** (3 - 2 * torch.arange(self.grad_steps+1, dtype=torch.float) / (self.grad_steps))
        epsilon_array = epsilon_array.unsqueeze(1) # .expand_dims(epsilon_array, axis=1)
        pi_explore = torch.cat([pi_array + epsilon_array * torch.randn(self.grad_steps+1, self.action_space) for _ in range(n_explore)], dim=0)
        return pi_explore

    def cone_explore(self, n_explore):
        alpha = math.pi / 4
        n = n_explore - 1
        pi = self.pi_net.pi.detach().cpu()
        d = self.get_grad().cpu()
        m = len(pi)
        pi = pi.unsqueeze(0)

        x = torch.FloatTensor(n, m).normal_()
        mag = torch.FloatTensor(n, 1).uniform_()

        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        d = d / (torch.norm(d) + 1e-8)

        cos = (x @ d).unsqueeze(1)

        dp = x - cos * d.unsqueeze(0)

        dp = dp / torch.norm(dp, dim=1, keepdim=True)

        acos = torch.acos(torch.clamp(torch.abs(cos), 0, 1-1e-8))

        new_cos = torch.cos(acos * alpha / (math.pi / 2))
        new_sin = torch.sin(acos * alpha / (math.pi / 2))

        cone = new_sin * dp + new_cos * d

        explore = pi - self.epsilon * mag * cone

        return torch.cat([pi, explore])

    def get_grad(self):

        self.pi_net.train()
        self.optimizer_pi.zero_grad()
        if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
            self.optimizer_derivative.zero_grad()
            grad = self.derivative_net(self.pi_net.pi.detach()).detach().squeeze(0)
            self.pi_net.grad_update(grad.clone())
        elif self.algorithm_method == 'value':
            self.optimizer_value.zero_grad()
            loss_pi = self.value_net(self.pi_net.pi)
            loss_pi.backward()
        elif self.algorithm_method == 'spline':
            self.spline_net.zero_grad()
            loss_pi = self.spline_net(self.pi_net.pi)
            loss_pi.backward()
        else:
            raise NotImplementedError

        if self.grad_clip != 0:
            nn.utils.clip_grad_norm_(self.pi_net.pi, self.grad_clip / self.pi_lr)

        self.pi_net.eval()
        grad = self.pi_net.pi.grad.detach().clone()
        return grad

    def pi_optimize(self):
        pi = self.pi_net.pi.detach().clone()
        pi_list = [pi]
        value_list = [self.step_policy(pi, to_env=False)]

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
