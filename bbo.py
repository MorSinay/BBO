import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from collections import defaultdict
from torchvision.utils import save_image
from environment import Env
from loguru import logger
from config import exp, args
from model import ValueNet, GradNet
from alg import Algorithm
import os
import copy

import itertools


class BBO(Algorithm):

    def __init__(self):
        super(BBO, self).__init__()

        self.env = Env(self.problem_index)
        self.pi_0 = self.env.get_initial_policy()

        if args.explore == 'grad_rand':
            self.explore_func = self.explore_grad_rand
        elif args.explore == 'grad_guided':
            self.explore_func = self.explore_grad_guided
        elif args.explore == 'grad_prop':
            self.explore_func = self.explore_grad_prop
        elif args.explore == 'rand':
            self.explore_func = self.explore_rand
        else:
            logger.info(f"explore: {args.explore}")
            raise NotImplementedError

        if self.grad:
            self.fit_func = self.fit_grad
            self.value_iter = 40
            self.pi_net = self.pi_0
            self.grad_net = GradNet()
            self.grad_net.to(self.device)
            self.optimizer_grad = torch.optim.Adam(self.grad_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=self.weight_decay)

        else:
            self.fit_func = self.fit_value
            self.value_iter = 4
            self.pi_net = nn.Parameter(self.pi_0)
            self.value_net = ValueNet()
            self.value_net.to(self.device)
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=self.weight_decay)

        self.optimizer_pi = torch.optim.SGD([self.pi_net], lr=self.pi_lr)
        self.q_loss = nn.SmoothL1Loss(reduction='none')

        self.replay_reward = torch.cuda.FloatTensor([])
        self.replay_policy = torch.cuda.FloatTensor([])

        self.mean = 0.
        self.std = 1.
        self.results = defaultdict(lambda: defaultdict(list))

    def update_pi_optimizer_lr(self):
        op_dict = self.optimizer_pi.state_dict()
        self.pi_lr *= 0.1
        self.pi_lr = max(self.pi_lr, 1e-6)
        op_dict['param_groups'][0]['lr'] = self.pi_lr
        self.optimizer_pi.load_state_dict(op_dict)

    def reset_pi(self):
        with torch.no_grad():
            if self.pi_net.grad is not None:
                self.pi_net.grad.zero_()
            self.pi_net.data = self.env.get_initial_policy()

    def learn(self):

        self.reset_pi()
        self.env.reset()
        self.results = defaultdict(lambda: defaultdict(list))

        self.results['image']['target'] = self.env.image_target.data.cpu()

        self.warmup()

        for n in tqdm(itertools.count(1)):

            pi_explore, reward_explore, images_explore = self.explore(self.batch)
            
            # results['aux']['pi_explore'].append(pi_explore)
            self.results['scalar']['reward_explore'].append(float(reward_explore.mean()))

            self.fit(pi_explore, reward_explore)

            self.optimize()

            pi = self.pi_net.detach()
            current_image, current_reward = self.env.evaluate(pi)

            self.results['scalar']['current_reward'].append(float(current_reward))

            if not n % self.train_epoch:

                self.results['histogram']['best_pi'] = self.env.best_policy.detach()
                self.results['histogram']['current'] = pi.detach()
                self.results['scalar']['best_reward'].append(float(self.env.best_reward))
                self.results['image']['best_image'] = self.env.best_image.detach().data.cpu()
                self.results['image']['current_image'] = current_image.detach().data.cpu()
                self.results['images']['images_explore'] = images_explore[:16].detach().data.cpu()
                yield self.results
                self.results = defaultdict(lambda: defaultdict(list))

    def norm(self, x):
        x = (x - self.mean) / (self.std + 1e-5)
        return x

    def denorm(self, x):
        x = (x * (self.std + 1e-5)) + self.mean
        return x

    def warmup(self):

        rewards = []
        explore_policies = []

        for _ in range(self.warm_up):
            pi_explore = torch.cuda.FloatTensor(self.batch, self.action_space).normal_()
            bbo_results = self.env.step(pi_explore)
            rewards.append(bbo_results.reward)
            explore_policies.append(pi_explore)
        rewards = torch.cat(rewards)
        explore_policies = torch.cat(explore_policies)

        self.mean = rewards.mean()
        self.std = rewards.std()
        self.results['scalar']['mean'].append(float(self.mean))
        self.results['scalar']['std'].append(float(self.std))

        best_idx = rewards.argmax()
        pi = explore_policies[best_idx]

        with torch.no_grad():
            self.pi_net.data = pi

        self.fit(explore_policies, rewards)

    def fit(self, pi_explore, reward_explore):

        self.replay_reward = torch.cat([self.replay_reward, self.norm(reward_explore)])[-self.replay_memory_size:]
        self.replay_policy = torch.cat([self.replay_policy, pi_explore])[-self.replay_memory_size:]
        self.fit_func()

    def fit_grad(self):

        len_replay = len(self.replay_policy)
        minibatches = len_replay // self.batch

        self.grad_net.train()
        for it in range(self.value_iter):

            shuffle_index = torch.randint(self.batch, (minibatches,)) + torch.arange(0, self.batch * minibatches,
                                                                                     self.batch)
            r_1 = self.replay_reward[shuffle_index].unsqueeze(1).repeat(1, self.batch)
            pi_1 = self.replay_policy[shuffle_index]

            r_2 = self.replay_reward.view(minibatches, self.batch)
            pi_2 = self.replay_policy.view(minibatches, self.batch, -1)
            pi_tag_1 = self.grad_net(pi_1).unsqueeze(1).repeat(1, self.batch, 1)
            pi_1 = pi_1.unsqueeze(1).repeat(1, self.batch, 1)

            if self.importance_sampling:
                w = torch.clamp((1 / (torch.norm(pi_2 - pi_1, p=2, dim=2) + 1e-4)).flatten(), 0, 1)
                w = w / w.max()
            else:
                w = 1.

            value = ((pi_2 - pi_1) * pi_tag_1).sum(dim=2).flatten()
            target = (r_2 - r_1).flatten()

            self.optimizer_grad.zero_grad()
            loss_q = (w * self.q_loss(value, target)).mean()
            loss_q.backward()
            self.optimizer_grad.step()

    def fit_value(self):

        len_replay = len(self.replay_policy)
        minibatches = len_replay // self.batch

        self.value_net.train()
        for it in range(self.value_iter):
            shuffle_indexes = np.random.choice(len_replay, (minibatches, self.batch), replace=True)
            for i in range(minibatches):
                samples = shuffle_indexes[i]
                r = self.replay_reward[samples]
                pi_explore = self.replay_policy[samples]

                self.optimizer_value.zero_grad()
                q_value = -self.value_net(pi_explore).view(-1)
                loss_q = self.q_loss(q_value, r).mean()
                loss_q.backward()
                self.optimizer_value.step()

                self.results['scalar']['value_loss'].append(float(loss_q))

    def get_n_grad_ahead(self, n):

        optimizer_state = copy.deepcopy(self.optimizer_pi.state_dict())
        pi_array = [self.pi_net.detach()]
        for _ in range(n-1):
            pi, _ = self.grad_step()
            pi_array.append(pi)

        self.optimizer_pi.load_state_dict(optimizer_state)
        with torch.no_grad():
            self.pi_net.data = pi_array[0]

        pi_array = torch.stack(pi_array)
        return pi_array

    def grad_step(self):
        if self.grad:
            self.grad_net.eval()
            grad = self.grad_net(self.pi_net).detach().squeeze(0)
            with torch.no_grad():
                self.pi_net.grad = -grad.detach()
        else:
            self.value_net.eval()
            self.optimizer_pi.zero_grad()
            loss_pi = self.value_net(self.pi_net)
            loss_pi.backward()

        nn.utils.clip_grad_norm_(self.pi_net, self.clip/self.pi_lr)
        self.optimizer_pi.step()

        pi = self.pi_net.detach()
        return pi, self.env.evaluate(pi).reward

    def explore_rand(self, n_explore):

            pi = self.pi_net.detach().unsqueeze(0)
            pi_explore = (1 - self.epsilon) * pi + self.epsilon * torch.cuda.FloatTensor(n_explore, self.action_space).normal_()
            # pi_explore = torch.cuda.FloatTensor(n_explore, self.action_space).normal_()

            return pi_explore

    def explore_grad_rand(self, n_explore):

        grads = self.get_grad().detach()
        pi = self.pi_net.detach().unsqueeze(0)

        explore_factor = self.delta * grads + self.epsilon * torch.cuda.FloatTensor(n_explore, self.action_space).normal_()
        explore_factor *= 0.9 ** (2 * torch.arange(n_explore, device=self.device, dtype=torch.float32)).reshape(n_explore, 1)

        pi_explore = pi + explore_factor  # gradient decent

        return pi_explore

    def explore_grad_guided(self, n_explore):

        pi_array = self.get_n_grad_ahead(self.grad_steps)
        n_explore = n_explore // self.grad_steps

        epsilon_array = 0.1 ** (3 - 2 * torch.arange(self.grad_steps, device=self.device, dtype=torch.float32)/(self.grad_steps - 1))
        epsilon_array = epsilon_array.unsqueeze(1)
        pi_explore = torch.cat([pi_array + epsilon_array * torch.cuda.FloatTensor(self.grad_steps, self.action_space).normal_()
                                for _ in range(n_explore)])

        return pi_explore

    def explore_grad_prop(self, n_explore):
        pi_array = self.get_n_grad_ahead(self.grad_steps + 1)
        n_explore = n_explore // self.grad_steps

        diff_pi = torch.norm(pi_array[1:] - pi_array[:-1], dim=1)

        epsilon_array = diff_pi * 10 ** (torch.arange(self.grad_steps, device=self.device, dtype=torch.float32) / (self.grad_steps - 1))
        epsilon_array = epsilon_array.unsqueeze(1)
        pi_explore = torch.cat([pi_array[:-1] + epsilon_array * torch.cuda.FloatTensor(n_explore, self.action_space).normal_()
                                for _ in range(n_explore)])

        return pi_explore

    def get_grad(self):
        if self.grad:
            self.grad_net.eval()
            grad = self.grad_net(self.pi_net).detach().squeeze(0)
            return grad
        else:
            self.value_net.eval()
            self.optimizer_pi.zero_grad()
            loss_pi = self.value_net(self.pi_net)
            loss_pi.backward()
            return self.pi_net.grad.detach()

    def optimize(self):

        pi = self.pi_net.detach()
        pi_list = [pi]
        value_list = [self.env.evaluate(pi).reward]

        for _ in range(self.grad_steps):
            pi, value = self.grad_step()
            pi_list.append(pi)
            value_list.append(value)

        if self.update_step == 'n_step':
            #no need to do action
            pass
        elif self.update_step == 'best_step':
            best_idx = np.argmin(value_list)
            with torch.no_grad():
                self.pi_net.data = pi_list[best_idx]
        elif self.update_step == 'first_vs_last':
            if value_list[-1] > value_list[0]:
                with torch.no_grad():
                    self.pi_net.data = pi_list[0]
        elif self.update_step == 'no_update':
            with torch.no_grad():
                self.pi_net.data = pi_list[0]
        else:
            raise NotImplementedError

    def explore(self, n_explore):

        pi_explore = self.explore_func(n_explore)
        bbo_results = self.env.step(pi_explore)
        rewards = bbo_results.reward
        images = bbo_results.image
        if self.best_explore_update:
            best_explore = rewards.argmax()
            with torch.no_grad():
                self.pi_net.data = pi_explore[best_explore]

        return pi_explore, rewards, images
