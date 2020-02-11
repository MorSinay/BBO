import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
import torch.nn as nn
from collections import defaultdict
from torchvision.utils import save_image
from config import args, DirsAndLocksSingleton
from model_ddpg import DuelNet, PiNet, SplineNet, MultipleOptimizer
import math
import os
import copy
import shutil

class Agent(object):

    def __init__(self, exp_name, env, checkpoint):
        self.cuda_id = args.cuda_default
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.action_space = args.action_space
        self.env = env
        self.dirs_locks = DirsAndLocksSingleton(exp_name)

        self.use_trust_region = args.trust_region
        self.batch = args.batch
        self.max_batch = args.batch
        self.n_explore = args.n_explore
        self.replay_memory_size = self.n_explore * args.replay_memory_factor
        self.problem_index = env.problem_iter
        self.value_lr = args.value_lr
        self.budget = args.budget
        self.checkpoint = checkpoint
        self.algorithm_method = args.algorithm
        self.grad_clip = args.grad_clip
        self.req_lambda = 1e-3
        self.divergence = 0
        self.best_explore_update = args.best_explore_update
        self.printing_interval = args.printing_interval
        self.analysis_dir = os.path.join(self.dirs_locks.analysis_dir, str(self.problem_index))
        if os.path.exists(self.analysis_dir):
            shutil.rmtree(self.analysis_dir, ignore_errors=True)
            os.makedirs(self.analysis_dir)
        else:
            os.makedirs(self.analysis_dir)

        self.frame = 0
        self.n_offset = 0
        self.results = defaultdict(list)
        self.tensor_replay_reward = torch.cuda.FloatTensor([])
        self.tensor_replay_policy = torch.cuda.FloatTensor([])
        self.pi_lr = args.pi_lr
        self.epsilon = args.epsilon * math.sqrt(self.action_space)
        self.delta = self.pi_lr
        self.warmup_minibatch = args.warmup_minibatch
        self.mean_grad = None
        self.alpha = args.alpha
        self.epsilon_factor = args.epsilon_factor
        self.spline = args.spline

        if args.explore == 'rand':
            self.exploration = self.exploration_rand
        elif args.explore == 'ball':
            self.exploration = self.ball_explore
        elif args.explore == 'cone':
            if self.action_space == 1:
                self.exploration = self.exploration_rand
            else:
                self.exploration = self.cone_explore_with_rand
                self.cone_angle = args.cone_angle
        else:
            print("explore:" + args.explore)
            raise NotImplementedError

        self.init = torch.FloatTensor(self.env.get_initial_solution()).to(self.device)
        self.pi_net = PiNet(self.init, self.device, self.action_space)
        self.optimizer_pi = torch.optim.SGD([self.pi_net.pi], lr=self.pi_lr)
        self.pi_net.eval()

        self.value_iter = args.learn_iteration
        if self.algorithm_method in ['EGL']:
            if self.spline:
                self.derivative_net = SplineNet(self.device, self.pi_net, output=self.action_space)
                self.derivative_net.to(self.device)
                # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
                opt_sparse = torch.optim.SparseAdam(self.derivative_net.embedding.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-04)
                opt_dense = torch.optim.Adam(self.derivative_net.head.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-04)
                self.optimizer_derivative = MultipleOptimizer(opt_sparse, opt_dense)
            else:
                self.derivative_net = DuelNet(self.pi_net, self.action_space)
                self.derivative_net.to(self.device)
                # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
                self.optimizer_derivative = torch.optim.Adam(self.derivative_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.derivative_net.eval()
            self.derivative_net_zero = copy.deepcopy(self.derivative_net.state_dict())
        elif self.algorithm_method == 'IGL':
            if self.spline:
                self.value_net = SplineNet(self.device, self.pi_net, output=1)
                self.value_net.to(self.device)
                # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
                opt_sparse = torch.optim.SparseAdam(self.value_net.embedding.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-04)
                opt_dense = torch.optim.Adam(self.value_net.head.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-04)
                self.optimizer_value = MultipleOptimizer(opt_sparse, opt_dense)
            else:
                self.value_net = DuelNet(self.pi_net, 1)
                self.value_net.to(self.device)
                # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
                self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)

            self.value_net.eval()
            self.value_net_zero = copy.deepcopy(self.value_net.state_dict())
        else:
            raise NotImplementedError

        if args.loss == 'huber':
            self.q_loss = nn.SmoothL1Loss(reduction='none')
        elif args.loss == 'mse':
            self.q_loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError

    def reset_result(self):
        self.results = defaultdict(list)

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
            elif k in ['rewards']:
                rewards = torch.cat(self.results[k]).cpu().numpy()
                np.save(path, rewards)
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
        if self.algorithm_method in ['EGL']:
            state = {'pi_net': self.pi_net.pi.detach(),
                     'derivative_net': self.derivative_net.state_dict(),
                     'optimizer_derivative': self.optimizer_derivative.state_dict(),
                     'optimizer_pi': self.optimizer_pi.state_dict(),
                     'aux': aux}
        elif self.algorithm_method == 'IGL':
            state = {'pi_net': self.pi_net.pi.detach(),
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
        if self.algorithm_method in ['EGL']:
            self.derivative_net.load_state_dict(state['derivative_net'])
            self.optimizer_derivative.load_state_dict(state['optimizer_derivative'])
        elif self.algorithm_method == 'IGL':
            self.value_net.load_state_dict(state['value_net'])
            self.optimizer_value.load_state_dict(state['optimizer_value'])
        else:
            raise NotImplementedError
        self.n_offset = state['aux']['n']

        return state['aux']

    def reset_net(self):
        if self.algorithm_method in ['EGL']:
            self.derivative_net.load_state_dict(self.derivative_net_zero)
            self.optimizer_derivative.state = defaultdict(dict)
        if self.algorithm_method in ['IGL']:
            self.value_net.load_state_dict(self.value_net_zero)
            self.optimizer_value.state = defaultdict(dict)

    def get_n_grad_ahead(self, n):

        optimizer_state = copy.deepcopy(self.optimizer_pi.state_dict())
        pi_array = [self.pi_net.pi.detach().clone()]
        for _ in range(n):
            pi, _ = self.get_grad(grad_step=True)
            pi_array.append(pi)

        self.optimizer_pi.load_state_dict(optimizer_state)
        self.pi_net.pi_update(pi_array[0].to(self.device))

        pi_array = torch.stack(pi_array)
        return pi_array

    def exploration_rand(self, n_explore):
        pi = self.pi_net.pi.detach().clone()
        rand_sign = (2*torch.randint(0, 2 ,size=(n_explore-1, self.action_space), device=self.device)-1).reshape(n_explore-1, self.action_space)
        pi_explore = pi - self.epsilon * rand_sign * torch.cuda.FloatTensor(n_explore-1, self.action_space).uniform_()
        return torch.cat([pi.unsqueeze(0), pi_explore], dim=0)

    def ball_explore_(self, pi, n_explore):
        pi = pi.unsqueeze(0)

        x = torch.cuda.FloatTensor(n_explore, self.action_space).normal_()
        mag = torch.cuda.FloatTensor(n_explore, 1).uniform_()

        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)

        explore = pi + self.epsilon * mag * x

        return explore

    def ball_explore(self, n_explore):
        pi = self.pi_net.pi.detach().clone()

        explore = self.ball_explore_(pi, n_explore-1)

        return torch.cat([pi.unsqueeze(0), explore], dim=0)

    def cone_explore(self, n_explore, angle, pi, grad):
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

        explore = pi - self.epsilon * mag * cone

        return explore

    def cone_explore_with_rand(self, n_explore):
        pi, grad = self.get_grad(grad_step=False)

        #explore_rand = self.cone_explore(n_explore//2, 1, pi, grad)
        explore_rand = self.ball_explore_(pi, n_explore//2)
        explore_cone = self.cone_explore(n_explore - n_explore // 2 - 1, self.cone_angle, pi, grad)

        return torch.cat([pi.unsqueeze(0), explore_rand, explore_cone], dim=0)


    def get_grad(self, grad_step=False):
        self.pi_net.train()
        self.optimizer_pi.zero_grad()
        if self.algorithm_method in ['EGL']:
            self.optimizer_derivative.zero_grad()
            grad = self.derivative_net(self.pi_net.pi).view_as(self.pi_net.pi).detach().clone()
            # replace NaN values with zeros
            grad[grad != grad] = 0
            self.pi_net.grad_update(grad)
        elif self.algorithm_method == 'IGL':
            self.optimizer_value.zero_grad()
            loss_pi = self.value_net(self.pi_net.pi)
            loss_pi.backward()
        else:
            raise NotImplementedError

        if self.grad_clip != 0:
            nn.utils.clip_grad_norm_(self.pi_net.pi, self.epsilon / self.pi_lr)

        if grad_step:
            self.optimizer_pi.step()

        self.pi_net.eval()
        pi = self.pi_net.pi.detach().clone()
        grad = self.pi_net.pi.grad.detach().clone()
        return pi, grad
