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

from config import consts, args, DirsAndLocksSingleton
from model_ddpg import DuelNet, DerivativeNet

import os
import copy

import itertools
mem_threshold = consts.mem_threshold

#TODO:
# vae change logvar to
# curl regularization
# ELAD - 1d visualizetion of bbo function - eval function and derivative using projection in a specific direction


class NPAgent(object):

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
        self.beta_lr = args.beta_lr
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
        self.lr_list = [1e-3, 1e-4, 1e-4]
        self.lr_index = 0

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

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.init = torch.tensor(self.env.get_initial_solution(), dtype=torch.float).to(self.device)
        if self.algorithm_method in ['first_order', 'second_order']:
            self.value_iter = 200
            self.beta_net = self.init
            self.derivative_net = DerivativeNet()
            self.derivative_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_derivative = torch.optim.Adam(self.derivative_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
        elif self.algorithm_method == 'value':
            self.value_iter = 20
            self.beta_net = nn.Parameter(self.init)
            self.value_net = DuelNet()
            self.value_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
        elif self.algorithm_method == 'anchor':
            self.value_iter = 100
            self.beta_net = self.init
            self.derivative_net = DerivativeNet()
            self.derivative_net.to(self.device)
            self.value_net = DuelNet()
            self.value_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_derivative = torch.optim.Adam(self.derivative_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
        else:
            raise NotImplementedError

        #self.optimizer_beta = torch.optim.SGD([self.beta_net], lr=self.beta_lr)
        self.optimizer_beta = torch.optim.SGD([self.beta_net], lr=self.lr_list[self.lr_index])
        self.q_loss = nn.SmoothL1Loss(reduction='none')

    def update_beta_optimizer_lr(self):
        op_dict = self.optimizer_beta.state_dict()
        self.lr_index = (self.lr_index + 1)%len(self.lr_list)
        self.lr_index = min(self.lr_index + 1 ,len(self.lr_list)-1)
        self.beta_lr = self.lr_list[self.lr_index]
        op_dict['param_groups'][0]['lr'] = self.beta_lr
        self.optimizer_beta.load_state_dict(op_dict)

    def save_results(self):
        for k in self.results.keys():
            path = os.path.join(self.analysis_dir, k +'.npy')
            if k in ['explore_policies', 'policies']:
                tmp = self.norm_beta(np.vstack(self.results[k]))
                if tmp is None:
                    assert False, "save_results"
                np.save(path, tmp)
            else:
                tmp = np.array(self.results[k]).flatten()
                if tmp is None:
                    assert False, "save_results"
                np.save(path, tmp)

        path = os.path.join(self.analysis_dir, 'f0.npy')
        np.save(path, self.env.get_f0())

        if self.action_space == 784:
            path = os.path.join(self.analysis_dir, 'reconstruction.png')
            save_image(self.beta_net.cpu().view(1, 28, 28), path)

    def reset_beta(self):
        with torch.no_grad():
            self.beta_net.data = torch.tensor(self.init, dtype=torch.float).to(self.device)

    def save_checkpoint(self, path, aux=None):
        if self.algorithm_method in ['first_order', 'second_order']:
            state = {'beta_net': self.beta_net,
                     'derivative_net': self.derivative_net.state_dict(),
                     'optimizer_derivative': self.optimizer_derivative.state_dict(),
                     'optimizer_beta': self.optimizer_beta.state_dict(),
                     'aux': aux}
        elif self.algorithm_method == 'value':
            state = {'beta_net': self.beta_net,
                     'value_net': self.value_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_beta': self.optimizer_beta.state_dict(),
                     'aux': aux}
        elif self.algorithm_method == 'anchor':
            state = {'beta_net': self.beta_net,
                     'derivative_net': self.derivative_net.state_dict(),
                     'optimizer_derivative': self.optimizer_derivative.state_dict(),
                     'value_net': self.value_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_beta': self.optimizer_beta.state_dict(),
                     'aux': aux}
        else:
            raise NotImplementedError

        torch.save(state, path)

    def norm_beta(self, policy):
        #policy = np.clip(policy, -1, 1)
        policy = np.tanh(policy)
        return policy

    def torch_norm_beta(self):
        return torch.tanh(self.beta_net)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            assert False, "load_checkpoint"
        state = torch.load(path, map_location="cuda:%d" % self.cuda_id)
        self.beta_net = state['beta_net'].to(self.device)
        self.optimizer_beta.load_state_dict(state['optimizer_beta'])
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
        #alpha = 0.1
        self.mean = mean
        self.std = std
        self.value_optimize()

    def warmup(self):
        #n_explore = self.warmup_minibatch*self.batch

        explore_policies = self.exploration_rand(2*self.batch)
        self.evaluate_step_policy(explore_policies)
        rewards = self.env.reward
        self.results['explore_policies'].append(explore_policies)
        self.results['rewards'].append(rewards)
        self.update_mean_std(rewards.mean(), rewards.std())

    def minimize(self, n_explore):

        self.reset_beta()
        self.env.reset()
        self.warmup()

        for self.frame in tqdm(itertools.count()):
            beta_explore, reward = self.exploration_step(n_explore)
            self.results['explore_policies'].append(beta_explore)
            self.results['rewards'].append(reward)

            if self.value_optimize():
                self.beta_optimize()
            else:
                continue

            self.save_checkpoint(self.checkpoint, {'n': self.frame})

            beta = self.beta_net.detach().data.cpu().numpy()
            beta_eval = self.evaluate_beta(beta)
            self.results['best_observed'].append(self.env.best_observed)
            self.results['beta_evaluate'].append(beta_eval)

            if self.bandage:
                self.bandage_update()

            self.results['policies'].append(beta)
            if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
                grad_norm = torch.norm(self.derivative_net(self.torch_norm_beta()).detach(), 2).detach().item()
                self.results['grad_norm'].append(grad_norm)
            if self.algorithm_method in ['value', 'anchor']:
                value = -self.value_net(self.torch_norm_beta()).data.cpu().numpy()
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

    def update_best_beta(self):
        rewards = np.hstack(self.results['rewards'])
        best_idx = rewards.argmin()
        beta = np.vstack(self.results['explore_policies'])[best_idx]
        with torch.no_grad():
            if self.action_space == 1:
                beta = float(beta)
            self.beta_net.data = torch.tensor(beta, dtype=torch.float).to(self.device)

    def bandage_update(self):

        replay_observed = np.hstack(self.results['rewards'])[-self.replay_memory_size:]
        replay_evaluate = np.array(self.results['beta_evaluate'])[-self.replay_memory_factor:]

        best_observed = self.results['best_observed'][-1]
        best_evaluate = np.min(self.results['beta_evaluate'])

        bst_bandage = best_observed < np.min(replay_observed)
        bte_bandage = best_evaluate < np.min(replay_evaluate)

        if bst_bandage or bte_bandage:
            self.update_best_beta()
            self.update_mean_std(replay_observed.mean(), replay_observed.std())
            self.update_beta_optimizer_lr()
            self.divergence += 1

    def value_optimize(self):
        replay_buffer_rewards = self.norm(np.hstack(self.results['rewards'])[-self.replay_memory_size:])
        replay_buffer_policy = np.vstack(self.results['explore_policies'])[-self.replay_memory_size:]

        len_replay_buffer = len(replay_buffer_rewards)
        minibatches = len_replay_buffer // self.batch

        self.tensor_replay_reward = torch.tensor(replay_buffer_rewards, dtype=torch.float).to(self.device, non_blocking=True)
        self.tensor_replay_policy = torch.tensor(self.norm_beta(replay_buffer_policy), dtype=torch.float).to(self.device, non_blocking=True)

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
            pi_1 = self.tensor_replay_policy[shuffle_index]
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

    def get_n_grad_ahead(self, n):

        optimizer_state = copy.deepcopy(self.optimizer_beta.state_dict())
        beta_array = [self.beta_net.detach().cpu().numpy()]
        for _ in range(n-1):
            beta, _ = self.grad_step()
            beta_array.append(beta)

        self.optimizer_beta.load_state_dict(optimizer_state)
        with torch.no_grad():
            self.beta_net.data = torch.tensor(beta_array[0], dtype=torch.float).to(self.device)

        beta_array = np.stack(beta_array)
        return beta_array

    def grad_step(self):

        if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
            self.derivative_net.eval()
            grad = self.derivative_net(self.torch_norm_beta()).detach().squeeze(0)
            with torch.no_grad():
                self.beta_net.grad = -grad.detach()  # /torch.norm(grad.detach(), 2)
        elif self.algorithm_method == 'value':
            self.value_net.eval()
            self.optimizer_beta.zero_grad()
            loss_beta = self.value_net(self.torch_norm_beta())
            loss_beta.backward()
        else:
            raise NotImplementedError

        if self.grad_clip == 0:
            nn.utils.clip_grad_norm_(self.pi_net.pi, self.grad_clip / self.beta_lr)

        self.optimizer_beta.step()
        beta = self.beta_net.detach().data.cpu().numpy()
        return beta, self.evaluate_beta(beta)

    def norm(self, x):
        x = (x - self.mean) / (self.std + 1e-5)
        #x -= self.mean
        return x

    def denorm(self, x):
        x = (x * (self.std + 1e-5)) + self.mean
        #x += self.mean
        return x

    def exploration_rand(self, n_explore):

            beta = self.beta_net.detach().data.cpu().numpy()
            beta_explore = beta + self.epsilon * np.random.randn(n_explore, self.action_space)
            return beta_explore

    def exploration_grad_rand(self, n_explore):

        grads = self.get_grad().cpu().numpy().copy()

        beta = self.beta_net.detach().data.cpu().numpy()

        explore_factor = self.delta * grads + self.epsilon * np.random.randn(n_explore, self.action_space)
        explore_factor *= 0.9 ** (2 * np.array(range(n_explore))).reshape(n_explore, 1)
        beta_explore = beta + explore_factor  # gradient decent
        return beta_explore

    def exploration_grad_direct(self, n_explore):
        beta_array = self.get_n_grad_ahead(self.grad_steps).reshape(self.grad_steps, self.action_space)
        n_explore = (n_explore//self.grad_steps)

        epsilon_array = 0.01 ** (3 - 2 * np.arange(self.grad_steps) / (self.grad_steps - 1))
        epsilon_array = np.expand_dims(epsilon_array, axis=1)
        beta_explore = np.concatenate([beta_array + epsilon_array * np.random.randn(self.grad_steps, self.action_space) for _ in range(n_explore)])

        return beta_explore

    def exploration_grad_direct_prop(self, n_explore):
        beta_array = self.get_n_grad_ahead(self.grad_steps + 1)
        n_explore = (n_explore // self.grad_steps)

        diff_beta = np.linalg.norm(beta_array[1:] - beta_array[:-1], axis=1)

        epsilon_array = diff_beta * 10 ** (np.arange(self.grad_steps) / (self.grad_steps - 1))
        epsilon_array = np.expand_dims(epsilon_array, axis=1)
        beta_explore = np.concatenate([beta_array[:-1] + epsilon_array * np.random.randn(self.grad_steps, self.action_space) for _ in range(n_explore)])

        return beta_explore

    def get_grad(self):

        if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
            self.derivative_net.eval()
            grad = self.derivative_net(self.torch_norm_beta()).detach().squeeze(0)
            return grad.detach()
        elif self.algorithm_method == 'value':
            self.value_net.eval()
            self.optimizer_beta.zero_grad()
            loss_beta = self.value_net(self.torch_norm_beta())
            loss_beta.backward()
            return self.beta_net.grad.detach()
        else:
            raise NotImplementedError

    def evaluate_beta(self, beta):
        return self.env.f(self.norm_beta(beta))

    def evaluate_step_policy(self, policy):
        self.env.step_policy(self.norm_beta(policy))

    def beta_optimize(self):
        beta = self.beta_net.detach().data.cpu().numpy()
        beta_list = [beta]
        value_list = [self.evaluate_beta(beta)]

        for _ in range(self.grad_steps):
            beta, value = self.grad_step()
            beta_list.append(beta)
            value_list.append(value)

        if self.update_step == 'n_step':
            #no need to do action
            pass
        elif self.update_step == 'best_step':
            best_idx = np.array(value_list).argmin()
            with torch.no_grad():
                self.beta_net.data = torch.tensor(beta_list[best_idx], dtype=torch.float).to(self.device)
        elif self.update_step == 'first_vs_last':
            if value_list[-1] > value_list[0]:
                with torch.no_grad():
                    self.beta_net.data = torch.tensor(beta_list[0], dtype=torch.float).to(self.device)
        elif self.update_step == 'no_update':
            with torch.no_grad():
                self.beta_net.data = torch.tensor(beta_list[0], dtype=torch.float).to(self.device)
        else:
            raise NotImplementedError

    def exploration_step(self, n_explore):
        beta_explore = self.exploration(n_explore)
        self.evaluate_step_policy(beta_explore)
        rewards = self.env.reward
        if self.best_explore_update:
            best_explore = rewards.argmax()
            with torch.no_grad():
                self.beta_net.data = torch.tensor(beta_explore[best_explore], dtype=torch.float).to(self.device)

        return beta_explore, rewards

