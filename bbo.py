import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from collections import defaultdict
from alg import Algorithm
from model import RobustNormalizer, TrustRegion

import itertools


class BBO(Algorithm):

    def __init__(self):
        super(BBO, self).__init__()

        self.tr = TrustRegion(self.pi_net)
        self.r_norm = RobustNormalizer(lr=0.3)

        self.best_pi = None
        self.best_reward = np.inf

        self.best_pi_score = self.evaluate(self.pi_net.detach()).reward
        self.no_change = 0

        self.replay_reward = torch.cuda.FloatTensor([])
        self.replay_policy = torch.cuda.FloatTensor([])

        self.results = defaultdict(lambda: defaultdict(list))

        if self.algorithm == 'egl':
            self.fit = self.fit_grad

        elif self.algorithm == 'igl':
            self.fit = self.fit_value

        else:
            raise NotImplementedError

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

    def evaluate(self, pi):

        pi = self.tr.unconstrained_to_real(pi)
        return self.env.evaluate(pi)

    def step(self, pi):
        pi = self.tr.unconstrained_to_real(pi)
        return self.env.step(pi)

    def optimize(self):

        self.reset_pi()
        self.env.reset()
        self.results = defaultdict(lambda: defaultdict(list))

        self.results['image']['target'] = self.env.image_target.data.cpu()

        self.warmup()

        for n in tqdm(itertools.count(1)):

            pi_explore, bbo_results = self.explore()

            reward_explore, image_explore, budget = bbo_results.reward, bbo_results.image, bbo_results.budget

            self.replay_reward = torch.cat([self.replay_reward, reward_explore])[-self.replay_memory_size:]
            self.replay_policy = torch.cat([self.replay_policy, pi_explore])[-self.replay_memory_size:]

            self.results['scalar']['reward_explore'].append(float(reward_explore.mean()))
            self.results['scalar']['budget'].append(float(budget.max()))

            self.fit()

            self.get_grad(grad_step=True)

            pi = self.pi_net.detach()
            bbo_results = self.evaluate(pi)

            current_image, current_reward = bbo_results.image, bbo_results.reward

            self.results['scalar']['current_reward'].append(float(current_reward))

            if not n % self.train_epoch:

                self.results['histogram']['best_pi'] = self.env.best_policy.detach()
                self.results['histogram']['current'] = pi.detach()
                self.results['scalar']['best_reward'].append(float(self.env.best_reward))
                self.results['image']['best_image'] = self.env.best_image.detach().data.cpu()
                self.results['image']['current_image'] = current_image.detach().data.cpu()
                self.results['images']['images_explore'] = image_explore[:16].detach().data.cpu()
                yield self.results
                self.results = defaultdict(lambda: defaultdict(list))

    def update_best_pi(self):
        pi = self.best_pi.detach().clone()
        self.tr.squeeze(pi)
        self.epsilon *= self.epsilon_factor
        self.pi_net.pi_update(self.tr.real_to_unconstrained(pi))

    def warmup(self):

        self.reset_net()
        self.r_norm.reset()

        pi_explore = self.exploration_rand(self.warmup_explore)

        bbo_results = self.step(pi_explore)
        reward_explore = bbo_results.reward

        best_explore = reward_explore.argmin()
        if self.best_reward > reward_explore[best_explore]:
            self.best_pi = self.tr.unconstrained_to_real(pi_explore[best_explore].detach().clone())
            self.best_reward = reward_explore[best_explore]

        self.r_norm(reward_explore, training=True)

        replay_size = (len(pi_explore) // self.n_explore) * self.n_explore

        pi_explore = pi_explore[-replay_size:]
        reward_explore = reward_explore[-replay_size:]

        self.replay_reward = torch.cat([self.replay_reward, reward_explore])[-self.replay_memory_size:]
        self.replay_policy = torch.cat([self.replay_policy, pi_explore])[-self.replay_memory_size:]

        self.fit()

    def fit_grad(self):

        len_replay_buffer = len(self.replay_reward)
        batch = min(self.batch, len_replay_buffer)
        minibatches = len_replay_buffer // batch

        replay_reward = self.r_norm(self.replay_reward, training=False)

        loss = 0

        self.grad_net.train()
        for _ in range(self.value_iter):
            anchor_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=False)
            batch_indexes = anchor_indexes // self.n_explore

            for i, anchor_index in enumerate(anchor_indexes):
                ref_index = torch.LongTensor(self.batch * batch_indexes[i][:, np.newaxis] + np.arange(self.batch)[np.newaxis, :])

                r_1 = replay_reward[anchor_index].unsqueeze(1).repeat(1, self.batch)
                r_2 = replay_reward[ref_index]
                pi_1 = self.replay_policy[anchor_index]
                pi_2 = self.replay_policy[ref_index]
                pi_tag_1 = self.grad_net(pi_1).unsqueeze(1).repeat(1, self.batch, 1)
                pi_1 = pi_1.unsqueeze(1).repeat(1, self.batch, 1)

                if self.importance_sampling:
                    w = torch.clamp((1 / (torch.norm(pi_2 - pi_1, p=2, dim=2) + 1e-4)).flatten(), 0, 1)
                else:
                    w = 1

                value = ((pi_2 - pi_1) * pi_tag_1).sum(dim=2).flatten()
                target = (r_2 - r_1).flatten()

                self.optimizer_grad.zero_grad()
                loss_q = (w * self.q_loss(value, target)).mean()
                loss += float(loss_q.detach())
                loss_q.backward()
                self.optimizer_grad.step()

        loss /= self.value_iter
        self.results['scalar']['grad_loss'] = loss
        self.grad_net.eval()

    def fit_value(self,):

        len_replay_buffer = len(self.replay_reward)
        batch = min(self.batch, len_replay_buffer)
        minibatches = len_replay_buffer // batch

        replay_reward = self.r_norm(self.replay_reward, training=False)

        loss = 0
        self.value_net.train()
        for _ in range(self.value_iter):
            shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, batch), replace=False)

            for i in range(minibatches):
                samples = shuffle_indexes[i]
                r = replay_reward[samples]
                pi_explore = self.replay_policy[samples]

                self.optimizer_value.zero_grad()
                q_value = self.value_net(pi_explore).flatten()
                loss_q = self.q_loss(q_value, r).mean()

                loss += float(loss_q.detach())
                loss_q.backward()
                self.optimizer_value.step()

        loss /= self.value_iter
        self.results['value_loss'].append(loss)
        self.value_net.eval()

    def explore(self):

        pi_explore = self.exploration(self.n_explore)

        bbo_results = self.step(pi_explore)

        rewards = bbo_results.reward
        self.r_norm(rewards, training=True)

        best_explore = rewards.argmin()
        if self.best_explore_update:
            self.pi_net.pi_update(pi_explore[best_explore])

        if self.best_reward > rewards[best_explore]:
            self.best_pi = self.tr.unconstrained_to_real(pi_explore[best_explore].detach().clone())
            self.best_reward = rewards[best_explore]

        return pi_explore, bbo_results
