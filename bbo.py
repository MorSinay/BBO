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
from loguru import logger

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
        self.tr_counter = 0

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

        self.env.reset()
        self.reset_net()
        self.reset_pi()

        self.results = defaultdict(lambda: defaultdict(list))
        self.results['image']['target'] = self.env.image_target.data.cpu().numpy()
        self.results['image']['init'] = self.evaluate(self.pi_net.pi).image.data.cpu().numpy()
        self.results['text']['attributes'] = self.env.attributes_target_text

        self.results['aux']['attributes'] = self.env.attributes_target.squeeze(0).cpu().numpy()
        self.results['aux']['problem'] = int(self.env.problem)
        self.results['aux']['landmarks'] = self.env.landmark_target.data.cpu().numpy()

        # self.results['pickle']['target'] = self.env.image_target.data.cpu().numpy()
        # self.results['pickle']['init'] = self.evaluate(self.pi_net.pi).image.data.cpu().numpy()

        self.warmup()
        counter = -1

        for n in tqdm(itertools.count(1)):
            counter += 1

            pi_explore, bbo_results = self.explore()

            reward_explore, image_explore, budget = bbo_results.reward, bbo_results.image, bbo_results.budget

            self.replay_reward = torch.cat([self.replay_reward, reward_explore])[-self.replay_memory_size:]
            self.replay_policy = torch.cat([self.replay_policy, pi_explore])[-self.replay_memory_size:]

            self.results['scalar']['reward_explore'].append(float(reward_explore.mean()))
            self.results['scalar']['budget'].append(float(budget.max()))

            self.fit()

            _, grad = self.get_grad(grad_step=True)

            self.results['scalar']['grad_norm'].append(float(torch.norm(grad) / (self.epsilon_factor ** self.tr_counter)))

            pi = self.pi_net.detach()
            bbo_results = self.evaluate(pi)

            current_image, current_reward = bbo_results.image, bbo_results.reward

            if current_reward < self.best_pi_score:
                self.no_change = 0
                self.best_pi_score = current_reward
            else:
                self.no_change += 1

            if current_reward < self.best_reward:
                self.best_reward = current_reward
                self.best_pi = self.tr.unconstrained_to_real(pi.clone())

            if counter > self.stop_con and self.no_change > (self.stop_con / 4):
                counter = 0
                self.tr_counter += 1
                self.reset_net()
                self.update_best_pi()
                logger.info("Update trust region")
                self.warmup()

            self.results['scalar']['current_reward'].append(float(current_reward))
            self.results['scalar']['replay_size'].append(len(self.replay_policy))

            if not n % self.train_epoch:

                # self.results['histogram']['best_pi'] = self.env.best_policy.detach()
                self.results['scalar']['tr_counter'].append(int(self.tr_counter))
                # self.results['histogram']['current'] = pi.detach()
                self.results['scalar']['best_reward'].append(float(self.env.best_reward))
                self.results['image']['best_image'] = self.env.best_image.detach().data.cpu().numpy()
                self.results['image']['current_image'] = current_image.detach().data.cpu().numpy()
                self.results['images']['images_explore'] = image_explore.detach().data.cpu().numpy()
                yield self.results

                if self.budget <= self.env.k.n:
                    break

                self.results = defaultdict(lambda: defaultdict(list))

    def update_best_pi(self):

        self.replay_policy = self.tr.unconstrained_to_real(self.replay_policy)
        pi = self.best_pi.detach().clone()
        self.tr.squeeze(pi)
        self.epsilon *= self.epsilon_factor
        self.pi_net.pi_update(self.tr.real_to_unconstrained(pi))
        self.replay_policy = self.tr.real_to_unconstrained(self.replay_policy)

    def warmup(self):

        self.r_norm.reset()

        for i in range(self.warmup_minibatch):

            # pi_explore = self.exploration_rand(self.n_explore)
            # bbo_results = self.step(pi_explore)
            # reward_explore = bbo_results.reward
            #
            # best_explore = reward_explore.argmin()
            # if self.best_reward > reward_explore[best_explore]:
            #     self.best_pi = self.tr.unconstrained_to_real(pi_explore[best_explore].detach().clone())
            #     self.best_reward = reward_explore[best_explore]
            #
            # self.r_norm(reward_explore, training=True)

            pi_explore, bbo_results = self.explore(func=self.ball_explore)
            reward_explore = bbo_results.reward

            self.replay_reward = torch.cat([self.replay_reward, reward_explore])[-self.replay_memory_size:]
            self.replay_policy = torch.cat([self.replay_policy, pi_explore])[-self.replay_memory_size:]

        self.fit()

    def fit_grad(self):

        len_replay_buffer = len(self.replay_reward)
        batch = min(self.batch, len_replay_buffer)
        minibatches = len_replay_buffer // batch

        replay_reward = self.r_norm(self.replay_reward, training=False)

        self.results['scalar']['r_max'].append(float(replay_reward.max()))
        self.results['scalar']['r_min'].append(float(replay_reward.min()))
        self.results['scalar']['r_avg'].append(float(replay_reward.mean()))

        loss = 0

        self.net.train()
        for _ in range(self.value_iter):
            anchor_indexes = np.random.choice(len_replay_buffer, (minibatches, batch), replace=False)
            ref_indexes = np.random.randint(0, self.n_explore, size=(minibatches, batch))
            explore_indexes = anchor_indexes // self.n_explore

            for i, anchor_index in enumerate(anchor_indexes):
                ref_index = torch.LongTensor(self.n_explore * explore_indexes[i] + ref_indexes[i])

                r_1 = replay_reward[anchor_index]
                r_2 = replay_reward[ref_index]
                pi_1 = self.replay_policy[anchor_index]
                pi_2 = self.replay_policy[ref_index]
                pi_tag_1 = self.net(pi_1)

                if self.importance_sampling:
                    w = torch.clamp((1 / (torch.norm(pi_2 - pi_1, p=2, dim=1) + 1e-4)).flatten(), 0, 1)
                else:
                    w = 1

                value = ((pi_2 - pi_1) * pi_tag_1).sum(dim=1)
                target = (r_2 - r_1)

                self.optimizer.zero_grad()
                self.optimizer_pi.zero_grad()
                loss_q = (w * self.q_loss(value, target)).mean()

                loss += float(loss_q.detach())
                loss_q.backward()
                self.optimizer.step()

        loss /= self.value_iter
        self.results['scalar']['grad_loss'] = loss

    def fit_value(self,):

        len_replay_buffer = len(self.replay_reward)
        batch = min(self.batch, len_replay_buffer)
        minibatches = len_replay_buffer // batch

        replay_reward = self.r_norm(self.replay_reward, training=False)

        loss = 0
        self.net.train()
        for _ in range(self.value_iter):
            shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, batch), replace=False)

            for i in range(minibatches):
                samples = shuffle_indexes[i]
                r = replay_reward[samples]
                pi_explore = self.replay_policy[samples]

                self.optimizer.zero_grad()
                q_value = self.net(pi_explore)
                loss_q = self.q_loss(q_value, r).mean()

                loss += float(loss_q.detach())
                loss_q.backward()
                self.optimizer.step()

        loss /= self.value_iter
        self.results['scalar']['value_loss'] = loss

    def explore(self, func=None):

        if func is None:
            pi_explore = self.exploration(self.n_explore)
        else:
            pi_explore = func(self.n_explore)

        bbo_results = self.step(pi_explore)

        rewards = bbo_results.reward
        self.r_norm(rewards, training=True)

        best_explore = rewards.argmin()
        if self.best_reward > rewards[best_explore]:
            self.best_pi = self.tr.unconstrained_to_real(pi_explore[best_explore].detach().clone())
            self.best_reward = rewards[best_explore]

        # rewards_scaled = self.r_norm(rewards)

        # if self.best_explore_update:
        #     soft_prob = torch.softmax(-rewards_scaled / (self.tr_counter + 1), dim=0).cpu().numpy()
        #     sampled_index = np.random.choice(len(soft_prob), p=soft_prob)
        #
        #     self.pi_net.pi_update(pi_explore[sampled_index])

        if self.best_explore_update:
            self.pi_net.pi_update(pi_explore[best_explore])

        return pi_explore, bbo_results
