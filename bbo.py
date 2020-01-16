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
from model import RobustNormalizer, TrustRegion

import itertools


class BBO(Algorithm):

    def __init__(self):
        super(BBO, self).__init__()

        self.pi_trust_region = TrustRegion(self.pi_net)
        self.r_norm = RobustNormalizer(lr=0.5)

        self.best_pi = None
        self.best_reward = np.inf

        self.best_pi_score = self.env.evaluate(self.pi_net.pi.detach())
        self.no_change = 0

        self.replay_reward = torch.cuda.FloatTensor([])
        self.replay_policy = torch.cuda.FloatTensor([])

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
            self.pi_net.data = self.env.get_initial_solution()

    def learn(self):

        self.reset_pi()
        self.env.reset()
        self.results = defaultdict(lambda: defaultdict(list))

        self.results['image']['target'] = self.env.image_target.data.cpu()

        self.warmup()

        for n in tqdm(itertools.count(1)):

            pi_explore, bbo_results = self.explore()

            reward_explore, image_explore, budget = bbo_results.reward, bbo_results.image, bbo_results.budget

            self.results['scalar']['reward_explore'].append(float(reward_explore.mean()))
            self.results['scalar']['budget'].append(float(budget.max()))

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

    def update_best_pi(self):
        pi = self.best_pi.detach().clone()
        self.pi_trust_region.squeeze(pi)
        self.epsilon *= self.epsilon_factor
        self.pi_net.pi_update(self.pi_trust_region.real_to_unconstrained(pi))

    def warmup(self):
        self.mean_grad = None
        self.reset_net()
        self.r_norm.reset()
        self.update_replay_buffer()
        self.value_optimize(self.value_iter)

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

    def explore(self):

        pi_explore = self.exploration(self.n_explore)
        bbo_results = self.env.step(pi_explore)
        rewards = bbo_results.reward

        if self.best_explore_update:
            best_explore = rewards.argmax()
            with torch.no_grad():
                self.pi_net.data = pi_explore[best_explore]

        return pi_explore, bbo_results
