import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from collections import defaultdict

from config import consts, args, DirsAndLocksSingleton
from model_ddpg import DuelNet, ResNet, DerivativeNet

from environment import Env
import os
import copy

import itertools
mem_threshold = consts.mem_threshold

class BBOAgent(object):

    def __init__(self, exp_name, env, checkpoint):

        #TODO:
        # Need to add upper_bound and lower bound and dim and initial_solution
        # Need to fix env class to support VAE
        # Update momentum to SGD < 0.9
        # Work on Graphs and compare between the algorithms
        # Compare with diffrent exploration - rand

        reward_str = "BBO"
        print("Learning POLICY method using {} with BBOAgent".format(reward_str))

        self.env = env
        self.dirs_locks = DirsAndLocksSingleton(exp_name)
        self.action_space = args.action_space
        self.epsilon = float(args.epsilon * self.action_space / (self.action_space - 1))
        self.delta = args.delta
        self.cuda_id = args.cuda_default
        self.batch = args.batch
        self.replay_memory_size = self.batch*args.replay_memory_factor
        self.problem_index = args.problem_index
        self.beta_lr = args.beta_lr
        self.value_lr = args.value_lr
        self.budget = args.budget * self.env.get_problem_dim()
        self.checkpoint = checkpoint
        self.use_grad_net = args.grad
        self.grad_steps = args.grad_steps
        self.stop_con = args.stop_con

        self.analysis_dir = os.path.join(self.dirs_locks.analysis_dir, str(self.problem_index))
        if not os.path.exists(self.analysis_dir):
            os.makedirs(self.analysis_dir)

        self.frame = 0
        self.n_offset = 0

        if args.explore == 'grad_rand':
            self.exploration = self.exploration_grad_rand
        elif args.explore == 'grad_direct':
            self.exploration = self.exploration_grad_direct
        elif args.explore == 'rand':
            self.exploration = self.exploration_rand
        else:
            raise NotImplementedError

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.init = torch.tensor(self.env.get_initial_solution(), dtype=torch.float).to(self.device)
        if self.use_grad_net:
            self.beta_net = self.init
            self.derivative_net = DerivativeNet()
            self.derivative_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_derivative = torch.optim.Adam(self.derivative_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            #self.optimizer_beta = torch.optim.SGD([self.beta_net], lr=self.beta_lr)
            #self.optimizer_beta = torch.optim.Adam([self.beta_net], lr=self.beta_lr, eps=1.5e-4, weight_decay=0)
        else:
            self.beta_net = nn.Parameter(self.init)
            self.value_net = DuelNet()
            self.value_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            #self.optimizer_beta = torch.optim.SGD([self.beta_net], lr=self.beta_lr)
            #self.optimizer_beta = torch.optim.Adam([self.beta_net], lr=self.beta_lr, eps=1.5e-4, weight_decay=0)

        if args.beta_optim == 'sgd':
            self.optimizer_beta = torch.optim.SGD([self.beta_net], lr=self.beta_lr)
        elif args.beta_optim == 'adam':
            self.optimizer_beta = torch.optim.Adam([self.beta_net], lr=self.beta_lr, eps=1.5e-4, weight_decay=0)
        else:
            raise NotImplementedError
        self.q_loss = nn.SmoothL1Loss(reduction='none')

    def save_results(self, results):
        os.path.join(self.dirs_locks.analysis_dir, str(self.problem_index))

        for k in results.keys():
            path = os.path.join(self.analysis_dir, k +'.npy')
            np.save(path, results[k])

    def reset_beta(self):
        self.beta_net.data = self.init

    def save_checkpoint(self, path, aux=None):
        if self.use_grad_net:
            state = {'beta_net': self.beta_net,
                     'derivative_net': self.derivative_net.state_dict(),
                     'optimizer_derivative': self.optimizer_derivative.state_dict(),
                     'aux': aux}
        else:
            state = {'beta_net': self.beta_net,
                     'value_net': self.value_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_beta': self.optimizer_beta.state_dict(),
                     'aux': aux}

        torch.save(state, path)

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            #return {'n':0}
            assert False, "load_checkpoint"

        state = torch.load(path, map_location="cuda:%d" % self.cuda_id)

        self.beta_net = state['beta_net'].to(self.device)
        if self.use_grad_net:
            self.derivative_net.load_state_dict(state['derivative_net'])
            self.optimizer_derivative.load_state_dict(state['optimizer_derivative'])
        else:
            self.value_net.load_state_dict(state['value_net'])
            self.optimizer_beta.load_state_dict(state['optimizer_beta'])
            self.optimizer_value.load_state_dict(state['optimizer_value'])

        self.n_offset = state['aux']['n']

        return state['aux']

    def find_min(self, n_explore):

        results = defaultdict(list)

        self.reset_beta()
        self.env.reset()

        self.value_net.eval()
        for self.frame in tqdm(itertools.count()):
            self.value_net.eval()

            beta_explore = self.exploration(n_explore)

            self.env.step_policy(beta_explore)

            results['explore_policies'].append(beta_explore)
            results['rewards'].append(self.env.reward)

            replay_buffer_rewards = np.hstack(results['rewards'])[-self.replay_memory_size:]
            replay_buffer_policy = np.vstack(results['explore_policies'])[-self.replay_memory_size:]
            len_replay_buffer = len(replay_buffer_rewards)
            minibatches = 20*(len_replay_buffer // self.batch)

            tensor_replay_reward = torch.tensor(self.norm(replay_buffer_rewards), dtype=torch.float).to(self.device, non_blocking=True)
            tensor_replay_policy = torch.tensor(replay_buffer_policy, dtype=torch.float).to(self.device, non_blocking=True)

            if minibatches < 2:
                continue

            self.value_net.train()
            for it in itertools.count():
                avg_loss = torch.tensor(0, dtype=torch.float)
                shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch),
                                                   replace=True)
                for i in range(minibatches):
                    samples = shuffle_indexes[i]
                    r = tensor_replay_reward[samples]
                    pi_explore = tensor_replay_policy[samples]

                    self.optimizer_value.zero_grad()
                    q_value = -self.value_net(pi_explore).view(-1)
                    loss_q = self.q_loss(q_value, r).mean()
                    avg_loss += loss_q.item()
                    loss_q.backward()
                    self.optimizer_value.step()

                avg_loss /= minibatches
                if it >= 1:
                    break

            self.value_net.eval()
            for _ in range(self.grad_steps):
                self.grad_step()

            self.save_checkpoint(self.checkpoint, {'n': self.frame})

            grads = self.beta_net.grad.detach().cpu().numpy()
            beta = self.beta_net.detach().data.cpu().numpy()

            results['grads'].append(grads)
            results['reward_mean'].append(replay_buffer_rewards.mean())
            results['policies'].append(beta)
            q_value = self.value_net(torch.tensor(beta_explore, dtype=torch.float).to(self.device))
            results['q_value'].append(q_value.data.cpu().numpy())
            results['best_observed'].append(self.env.best_observed)
            results['beta_evaluate'].append(self.env.f(beta))
            results['ts'].append(self.env.t)
            results['q_loss'].append(avg_loss.numpy())

            yield results

            if len(results['best_observed']) > self.stop_con and results['best_observed'][-1] == results['best_observed'][-self.stop_con]:
                self.save_checkpoint(self.checkpoint, {'n': self.frame})
                self.save_results(results)
                print("VALUE IS NOT CHANGING - FRAME %d" % self.frame)
                break

            if results['ts'][-1]:
                self.save_checkpoint(self.checkpoint, {'n': self.frame})
                self.save_results(results)
                print("FINISHED SUCCESSFULLY - FRAME %d" % self.frame)
                break

            if self.frame >= self.budget:
                self.save_results(results)
                print("FAILED")
                break

    # def find_min_temp(self, n_explore):
    #
    #     results = defaultdict(list)
    #
    #     self.reset_beta()
    #     self.env.reset()
    #
    #     self.value_net.eval()
    #     self.optimizer_beta.zero_grad()
    #     loss_beta = -self.value_net(self.beta_net)
    #     loss_beta.backward()
    #
    #     grads = self.beta_net.grad.detach().cpu().numpy().copy()
    #
    #     for self.frame in tqdm(itertools.count()):
    #         self.value_net.eval()
    #
    #         beta = self.beta_net.detach().data.cpu().numpy()
    #
    #         explore_factor = self.delta * grads + self.epsilon * np.random.randn(n_explore, self.action_space)
    #         explore_factor *= 0.9 ** (2 * np.array(range(n_explore))).reshape(n_explore, 1)
    #         beta_explore = beta + explore_factor #gradient decent
    #
    #         lower_bounds, upper_bounds = self.env.constrains()
    #         beta_explore = np.clip(beta_explore, lower_bounds, upper_bounds)
    #
    #         self.env.step_policy(beta_explore)
    #
    #         results['explore_policies'].append(beta_explore)
    #         results['rewards'].append(self.env.reward)
    #
    #         replay_buffer_rewards = np.hstack(results['rewards'])[-self.replay_memory_size:]
    #         replay_buffer_policy = np.vstack(results['explore_policies'])[-self.replay_memory_size:]
    #         len_replay_buffer = len(replay_buffer_rewards)
    #         minibatches = 20 * (len_replay_buffer // self.batch)
    #
    #         tensor_replay_reward = torch.tensor(self.norm(replay_buffer_rewards), dtype=torch.float).to(self.device, non_blocking=True)
    #         tensor_replay_policy = torch.tensor(replay_buffer_policy, dtype=torch.float).to(self.device, non_blocking=True)
    #
    #         if minibatches < 2:
    #             continue
    #
    #         self.value_net.train()
    #         for it in itertools.count():
    #             avg_loss = torch.tensor(0, dtype=torch.float)
    #             shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch),
    #                                                replace=True)
    #             for i in range(minibatches):
    #                 samples = shuffle_indexes[i]
    #                 r = tensor_replay_reward[samples]
    #                 pi_explore = tensor_replay_policy[samples]
    #
    #                 self.optimizer_value.zero_grad()
    #                 q_value = -self.value_net(pi_explore).view(-1)
    #                 loss_q = self.q_loss(q_value, r).mean()
    #                 avg_loss += loss_q.item()
    #                 loss_q.backward()
    #                 self.optimizer_value.step()
    #
    #             avg_loss /= minibatches
    #             if it >= 1:
    #                 break
    #
    #         self.value_net.eval()
    #         for _ in range(self.grad_steps):
    #             self.optimizer_beta.zero_grad()
    #             loss_beta = self.value_net(self.beta_net)
    #             loss_beta.backward()
    #             self.optimizer_beta.step()
    #
    #         self.save_checkpoint(self.checkpoint, {'n': self.frame})
    #
    #         grads = self.beta_net.grad.detach().cpu().numpy()
    #
    #         results['grads'].append(grads)
    #         results['reward_mean'].append(replay_buffer_rewards.mean())
    #         results['policies'].append(beta)
    #         q_value = self.value_net(torch.tensor(beta_explore, dtype=torch.float).to(self.device))
    #         results['q_value'].append(q_value.data.cpu().numpy())
    #         results['best_observed'].append(self.env.best_observed)
    #         results['beta_evaluate'].append(self.env.f(beta))
    #         results['ts'].append(self.env.t)
    #         results['q_loss'].append(avg_loss.numpy())
    #
    #         yield results
    #
    #         if len(results['best_observed']) > self.stop_con and results['best_observed'][-1] == results['best_observed'][-self.stop_con]:
    #             self.save_checkpoint(self.checkpoint, {'n': self.frame})
    #             print("VALUE IS NOT CHANGING - FRAME %d" % self.frame)
    #             self.save_results(results)
    #             break
    #
    #         if results['ts'][-1]:
    #             self.save_checkpoint(self.checkpoint, {'n': self.frame})
    #             print("FINISHED SUCCESSFULLY - FRAME %d" % self.frame)
    #             self.save_results(results)
    #             break
    #
    #         if self.frame >= self.budget:
    #             print("FAILED")
    #             self.save_results(results)
    #             break

    def find_min_grad_eval(self, n_explore):

        results = defaultdict(list)

        self.reset_beta()
        self.env.reset()

        self.derivative_net.eval()

        for self.frame in tqdm(itertools.count()):
            self.derivative_net.eval()

            beta_explore = self.exploration(n_explore)

            self.env.step_policy(beta_explore)

            results['explore_policies'].append(beta_explore)
            results['rewards'].append(self.env.reward)

            replay_buffer_rewards = np.hstack(results['rewards'])[-self.replay_memory_size:]
            replay_buffer_policy = np.vstack(results['explore_policies'])[-self.replay_memory_size:]
            len_replay_buffer = len(replay_buffer_rewards)
            minibatches = len_replay_buffer // self.batch

            tensor_replay_reward = torch.tensor(self.norm(replay_buffer_rewards), dtype=torch.float).to(self.device, non_blocking=True)
            tensor_replay_policy = torch.tensor(replay_buffer_policy, dtype=torch.float).to(self.device, non_blocking=True)

            if minibatches < 2:
                continue

            for it in itertools.count():
                shuffle_index = np.random.randint(self.batch)

                r_1 = tensor_replay_reward[shuffle_index::self.batch].unsqueeze(1).repeat(1, self.batch)
                r_2 = tensor_replay_reward.view(minibatches, self.batch)
                pi_1 = tensor_replay_policy[shuffle_index::self.batch]
                pi_tag_1 = self.derivative_net(pi_1).unsqueeze(1).repeat(1, self.batch, 1)
                pi_1 = pi_1.unsqueeze(1).repeat(1, self.batch, 1)
                pi_2 = tensor_replay_policy.view(minibatches, self.batch, -1)
                w = 1 # (1 / (torch.norm(pi_2-pi_1, p=2, dim=2) + 1e-4)).flatten()

                value = ((pi_2 - pi_1) * pi_tag_1).sum(dim=2).flatten()
                target = (r_2 - r_1).flatten()

                self.optimizer_derivative.zero_grad()
                loss_q = (w*self.q_loss(value, target)).mean()
                loss = loss_q.item()
                loss_q.backward()
                self.optimizer_derivative.step()

                if it >= 200:
                    break

            self.derivative_net.eval()
            for _ in range(self.grad_steps):
                self.grad_step()

            self.save_checkpoint(self.checkpoint, {'n': self.frame})

            grads = self.beta_net.grad.detach().cpu().numpy().copy()
            beta = self.beta_net.detach().data.cpu().numpy()

            results['grads'].append(grads)
            results['policies'].append(beta)
            results['best_observed'].append(self.env.best_observed)
            results['beta_evaluate'].append(self.env.f(beta))
            results['ts'].append(self.env.t)
            results['q_loss'].append(loss)
            results['value'].append(float(value.mean()))
            results['target'].append(float(target.mean()))

            yield results

            if len(results['best_observed']) > self.stop_con and results['best_observed'][-1] == results['best_observed'][-self.stop_con]:
                self.save_checkpoint(self.checkpoint, {'n': self.frame})
                print("VALUE IS NOT CHANGING - FRAME %d" % self.frame)
                self.save_results(results)
                break

            if results['ts'][-1]:
                self.save_checkpoint(self.checkpoint, {'n': self.frame})
                print("FINISHED SUCCESSFULLY - FRAME %d" % self.frame)
                self.save_results(results)
                break

            if self.frame >= self.budget:
                print("FAILED")
                self.save_results(results)
                break

    def get_n_grad_ahead(self, n):

        optimizer_state = copy.deepcopy(self.optimizer_beta.state_dict())
        beta = self.beta_net.detach()

        beta_array = [self.beta_net.detach().cpu().numpy()]
        for _ in range(n-1):
            # self.optimizer_beta.zero_grad()
            # loss_beta = self.value_net(self.beta_net)
            # loss_beta.backward()
            # self.optimizer_beta.step()
            self.grad_step()
            beta_array.append(self.beta_net.detach().cpu().numpy())

        self.optimizer_beta.load_state_dict(optimizer_state)
        with torch.no_grad():
            self.beta_net.data = beta.data

        beta_array = np.stack(beta_array)
        return beta_array

    def grad_step(self):
        if self.use_grad_net:
            grad = self.derivative_net(self.beta_net).detach().squeeze(0)
            with torch.no_grad():
                self.beta_net.grad = -grad.detach()
        else:
            self.optimizer_beta.zero_grad()
            loss_beta = self.value_net(self.beta_net)
            loss_beta.backward()

        self.optimizer_beta.step()

        with torch.no_grad():
            beta = self.beta_net.detach().data.cpu().numpy()
            lower_bounds, upper_bounds = self.env.constrains()
            beta = np.clip(beta, lower_bounds, upper_bounds)
            self.beta_net.data = torch.tensor(beta, dtype=torch.float).to(self.device)

    def norm(self, x):
        mean = x.mean()
        x = x - mean
        return x

    def exploration_rand(self, n_explore):

            beta = self.beta_net.detach().data.cpu().numpy()
            beta_explore = beta + self.epsilon * np.random.randn(n_explore, self.action_space)
            lower_bounds, upper_bounds = self.env.constrains()
            beta_explore = np.clip(beta_explore, lower_bounds, upper_bounds)

            return beta_explore

    def exploration_grad_rand(self, n_explore):

        grads = self.get_grad().cpu().numpy().copy()

        beta = self.beta_net.detach().data.cpu().numpy()
#        beta = np.clip(beta, lower_bounds, upper_bounds)

        explore_factor = self.delta * grads + self.epsilon * np.random.randn(n_explore, self.action_space)
        explore_factor *= 0.9 ** (2 * np.array(range(n_explore))).reshape(n_explore, 1)
        beta_explore = beta + explore_factor  # gradient decent
        lower_bounds, upper_bounds = self.env.constrains()
        beta_explore = np.clip(beta_explore, lower_bounds, upper_bounds)

        return beta_explore

    def exploration_grad_direct(self, n_explore):
        beta_array = self.get_n_grad_ahead(self.grad_steps)
        n_explore = (n_explore//self.grad_steps)

        epsilon_array = 0.1 ** (3 - 2*np.arange(self.grad_steps)/(self.grad_steps - 1))
        epsilon_array = np.expand_dims(epsilon_array, axis=1)
        beta_explore = np.concatenate([beta_array + epsilon_array * np.random.randn(self.grad_steps, self.action_space) for _ in range(n_explore)])
        lower_bounds, upper_bounds = self.env.constrains()
        beta_explore = np.clip(beta_explore, lower_bounds, upper_bounds)

        return beta_explore

    def get_grad(self):
        if self.use_grad_net:
            grad = self.derivative_net(self.beta_net).detach().squeeze(0)
            return grad.detach()
        else:
            self.optimizer_beta.zero_grad()
            loss_beta = self.value_net(self.beta_net)
            loss_beta.backward()
            return self.beta_net.grad.detach()

