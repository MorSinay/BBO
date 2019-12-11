import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
import torch.autograd as autograd
from agent import Agent

from config import consts

import itertools
mem_threshold = consts.mem_threshold

#TODO:
# vae change logvar to
# curl regularization
# ELAD - 1d visualizetion of bbo function - eval function and derivative using projection in a specific direction


class BBOAgent(Agent):

    def __init__(self, exp_name, env, checkpoint):
        super(BBOAgent, self).__init__(exp_name, env, checkpoint)

    def minimize(self, n_explore):

        self.env.reset()
        self.warmup()
        self.pi_net.eval()
        for self.frame in tqdm(itertools.count()):
            pi_explore, reward = self.exploration_step(n_explore)
            self.results['explore_policies'].append(pi_explore)
            self.results['rewards'].append(reward)

            if self.value_optimize():
                self.pi_optimize()
            else:
                continue

            self.save_checkpoint(self.checkpoint, {'n': self.frame})

            pi = self.pi_net.pi.detach().clone()
            pi_eval = self.f_policy(pi)
            self.results['best_observed'].append(self.env.best_observed)
            self.results['pi_evaluate'].append(pi_eval)

            if self.bandage:
                self.bandage_update()

            self.results['policies'].append(pi)
            self.pi_net.eval()
            if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
                grad_norm = torch.norm(self.derivative_net(self.pi_net()).detach(), 2).detach().item()
                self.results['grad_norm'].append(grad_norm)
            if self.algorithm_method in ['value', 'anchor']:
                value = -self.value_net(self.pi_net()).detach().item()
                self.results['value'].append(self.denorm(value))

            self.results['ts'].append(self.env.t)
            self.results['divergence'].append(self.divergence)

            yield self.results

            if self.results['ts'][-1]:
                self.save_checkpoint(self.checkpoint, {'n': self.frame})
                self.save_results()
                print("FINISHED SUCCESSFULLY - FRAME %d" % self.frame)
                break

            if len(self.results['best_observed']) > self.stop_con and self.results['best_observed'][-1] == self.results['best_observed'][-self.stop_con]:
                self.save_checkpoint(self.checkpoint, {'n': self.frame})
                self.save_results()
                print("VALUE IS NOT CHANGING - FRAME %d" % self.frame)
                break

            if self.frame >= self.budget:
                self.save_results()
                print("FAILED")
                break

    def anchor_method_optimize(self, len_replay_buffer, minibatches):
        self.value_method_optimize(len_replay_buffer, minibatches)

        loss = 0
        self.derivative_net.train()
        self.value_net.eval()
        for it in itertools.count():
            shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=True)
            for i in range(minibatches):
                samples = shuffle_indexes[i]
                pi_1 = self.tensor_replay_policy[samples]
                pi_tag_1 = self.derivative_net(pi_1)
                pi_2 = pi_1.detach() + torch.randn(self.batch, self.action_space).to(self.device, non_blocking=True)

                r_1 = self.tensor_replay_reward[samples]
                r_2 = -self.value_net(pi_2).squeeze(1).detach()

                if self.importance_sampling:
                    w = torch.clamp((1 / (torch.norm(pi_2 - pi_1, p=2, dim=1) + 1e-4)).flatten(), 0, 1)
                else:
                    w = 1

                value = ((pi_2 - pi_1) * pi_tag_1).sum(dim=1)
                target = (r_2 - r_1)

                self.optimizer_derivative.zero_grad()
                loss_q = (w * self.q_loss(value, target)).mean()
                loss += loss_q.detach().item()
                loss_q.backward()
                self.optimizer_derivative.step()

            if it >= self.value_iter:
                break

        loss /= self.value_iter
        self.results['derivative_loss'].append(loss)

    def value_method_optimize(self, len_replay_buffer, minibatches):
        loss = 0
        self.value_net.train()
        for it in itertools.count():
            shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=True)
            for i in range(minibatches):
                samples = shuffle_indexes[i]
                r = self.tensor_replay_reward[samples]
                pi_explore = self.tensor_replay_policy[samples]

                self.optimizer_value.zero_grad()
                q_value = -self.value_net(pi_explore).view(-1)
                loss_q = self.q_loss(q_value, r).mean()
                loss += loss_q.detach().item()
                loss_q.backward()
                self.optimizer_value.step()

            if it >= self.value_iter:
                break

        loss /= self.value_iter
        self.results['value_loss'].append(loss)

    def first_order_method_optimize(self, len_replay_buffer, minibatches):

        loss = 0
        self.derivative_net.train()
        for it in itertools.count():
            shuffle_index = np.random.randint(0, self.batch, size=(minibatches,)) + np.arange(0, self.batch * minibatches, self.batch)
            r_1 = self.tensor_replay_reward[shuffle_index].unsqueeze(1).repeat(1, self.batch)
            r_2 = self.tensor_replay_reward.view(minibatches, self.batch)
            pi_1 = self.tensor_replay_policy[shuffle_index]
            pi_2 = self.tensor_replay_policy.view(minibatches, self.batch, -1)
            pi_tag_1 = self.derivative_net(pi_1).unsqueeze(1).repeat(1, self.batch, 1)
            pi_1 = pi_1.unsqueeze(1).repeat(1, self.batch, 1)

            if self.importance_sampling:
                w = torch.clamp((1 / (torch.norm(pi_2 - pi_1, p=2, dim=2) + 1e-4)).flatten(), 0, 1)
            else:
                w = 1

            value = ((pi_2 - pi_1) * pi_tag_1).sum(dim=2).flatten()
            target = (r_2 - r_1).flatten()

            self.optimizer_derivative.zero_grad()
            loss_q = (w * self.q_loss(value, target)).mean()
            loss += loss_q.detach().item()
            loss_q.backward()
            self.optimizer_derivative.step()

            if it >= self.value_iter:
                break

        loss /= self.value_iter
        self.results['derivative_loss'].append(loss)


    def second_order_method_optimize(self, len_replay_buffer, minibatches):
        loss = 0
        mid_val = True

        self.derivative_net.train()
        for it in itertools.count():

            shuffle_index = np.random.randint(0, self.batch, size=(minibatches,)) + np.arange(0, self.batch * minibatches, self.batch)
            r_1 = self.tensor_replay_reward[shuffle_index].unsqueeze(1).repeat(1, self.batch)
            r_2 = self.tensor_replay_reward.view(minibatches, self.batch)
            pi_1 = self.tensor_replay_policy[shuffle_index].unsqueeze(1).repeat(1, self.batch, 1).reshape(-1,self.action_space)
            pi_2 = self.tensor_replay_policy.view(minibatches * self.batch, -1)
            delta_pi = pi_2 - pi_1
            if mid_val:
                mid_pi = (pi_1 + pi_2) / 2
                mid_pi = autograd.Variable(mid_pi, requires_grad=True)
                pi_tag_mid = self.derivative_net(mid_pi)
                pi_tag_1 = self.derivative_net(pi_1)
                pi_tag_mid_dot_delta = (pi_tag_mid * delta_pi).sum(dim=1)
                gradients = autograd.grad(outputs=pi_tag_mid_dot_delta, inputs=mid_pi, grad_outputs=torch.cuda.FloatTensor(pi_tag_mid_dot_delta.size()).fill_(1.),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0].detach()
                second_order = 0.5 * (delta_pi * gradients.detach()).sum(dim=1)
            else:
                pi_1 = autograd.Variable(pi_1, requires_grad=True)
                pi_tag_1 = self.derivative_net(pi_1)
                pi_tag_1_dot_delta = (pi_tag_1 * delta_pi).sum(dim=1)
                gradients = autograd.grad(outputs=pi_tag_1_dot_delta, inputs=pi_1, grad_outputs=torch.cuda.FloatTensor(pi_tag_1_dot_delta.size()).fill_(1.),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0].detach()
                second_order = 0.5 * (delta_pi * gradients.detach()).sum(dim=1)

            if self.importance_sampling:
                w = torch.clamp((1 / (torch.norm(pi_2 - pi_1, p=2, dim=1) + 1e-4)).flatten(), 0, 1)
            else:
                w = 1

            value = (delta_pi * pi_tag_1).sum(dim=1).flatten()
            target = (r_2 - r_1).flatten() - second_order

            self.optimizer_derivative.zero_grad()
            loss_q = (w * self.q_loss(value, target)).mean()
            loss += loss_q.detach().item()
            loss_q.backward()
            self.optimizer_derivative.step()

            if it >= self.value_iter:
                break

        loss /= self.value_iter
        self.results['derivative_loss'].append(loss)

    def value_optimize(self):
        self.pi_net.eval()
        replay_buffer_rewards = self.norm(np.hstack(self.results['rewards'])[-self.replay_memory_size:])
        replay_buffer_policy = torch.cat(self.results['explore_policies'], dim=0)[-self.replay_memory_size:]

        len_replay_buffer = len(replay_buffer_rewards)
        minibatches = len_replay_buffer // self.batch

        self.tensor_replay_reward = torch.tensor(replay_buffer_rewards, dtype=torch.float).to(self.device, non_blocking=True)
        self.tensor_replay_policy = self.pi_net(replay_buffer_policy.to(self.device))

        if minibatches < self.warmup_minibatch:
            return False

        if self.algorithm_method == 'first_order':
            self.first_order_method_optimize(len_replay_buffer, minibatches)
        elif self.algorithm_method == 'value':
            self.value_method_optimize(len_replay_buffer, minibatches)
        elif self.algorithm_method == 'second_order':
            self.second_order_method_optimize(len_replay_buffer, minibatches)
        elif self.algorithm_method == 'anchor':
            self.anchor_method_optimize(len_replay_buffer, minibatches)
        else:
            raise NotImplementedError
        return True

    def bandage_update(self):

        replay_observed = np.hstack(self.results['rewards'])[-self.replay_memory_size:]
        replay_evaluate = -np.array(self.results['pi_evaluate'])[-self.replay_memory_factor:]

        best_observed = -self.results['best_observed'][-1]
        best_evaluate = -np.max(self.results['pi_evaluate'])

        bst_bandage = best_observed > np.max(replay_observed)
        #bte_bandage = best_evaluate > np.max(replay_evaluate)

        if bst_bandage: #or bte_bandage:
            self.update_best_pi()
            self.update_mean_std(replay_observed.mean(), replay_observed.std())
            self.update_pi_optimizer_lr()
            self.divergence += 1


