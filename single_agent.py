import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from collections import defaultdict

from config import consts, args, DirsAndLocksSingleton
from model_ddpg import DuelNet, ResNet

from environment import Env
import os

import itertools
mem_threshold = consts.mem_threshold


class BBOAgent(object):

    def __init__(self, exp_name, problem, checkpoint):

        reward_str = "BBO"
        print("Learning POLICY method using {} with BBOAgent".format(reward_str))

        self.problem = problem
        self.dirs_locks = DirsAndLocksSingleton(exp_name)
        self.action_space = args.action_space
        self.epsilon = float(args.epsilon * self.action_space / (self.action_space - 1))
        self.delta = args.delta
        self.cuda_id = args.cuda_default
        self.batch = args.batch
        self.replay_memory_size = args.replay_memory_size
        self.problem_index = args.problem_index
        self.beta_lr = args.beta_lr
        self.value_lr = args.value_lr
        self.budget = args.budget * self.problem.dimension
        self.checkpoint = checkpoint

        self.env = Env(self.problem)
        self.frame = 0
        self.n_offset = 0

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.init = torch.tensor(self.problem.initial_solution, dtype=torch.float).to(self.device)
        self.beta_net = nn.Parameter(self.init)
        self.value_net = DuelNet()
        self.value_net.to(self.device)
        self.q_loss = nn.SmoothL1Loss(reduction='none')

        # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
        #self.optimizer_value = torch.optim.SGD(self.value_net.parameters(), lr=self.value_lr)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)

        self.optimizer_beta = torch.optim.Adam([self.beta_net], lr=self.beta_lr, eps=1.5e-4, weight_decay=0)
        # self.optimizer_beta = torch.optim.Adam([self.beta_net], lr=0.00025/4, eps=1.5e-4, weight_decay=0)

    def reset_beta(self):
        self.beta_net.data = self.init

    def save_checkpoint(self, path, aux=None):

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
        self.optimizer_beta.zero_grad()
        loss_beta = -self.value_net(self.beta_net)
        loss_beta.backward()

        grads = self.beta_net.grad.detach().cpu().numpy().copy()

        for self.frame in tqdm(itertools.count()):
            self.value_net.eval()
            beta = self.beta_net.detach().data.cpu().numpy()
            beta = np.clip(beta, self.problem.lower_bounds, self.problem.upper_bounds)

            explore_factor = self.delta * grads + self.epsilon * np.random.randn(n_explore, self.action_space)
            explore_factor *= 0.9 ** (2 * np.array(range(n_explore))).reshape(n_explore, 1)
            beta_explore = beta + explore_factor #gradient decent
            beta_explore = np.clip(beta_explore, self.problem.lower_bounds, self.problem.upper_bounds)

            self.env.step_policy(beta_explore)

            results['explore_policies'].append(beta_explore)
            results['rewards'].append(self.env.reward)

            replay_buffer_rewards = np.hstack(results['rewards'])[-self.replay_memory_size:]
            replay_buffer_policy = np.vstack(results['explore_policies'])[-self.replay_memory_size:]
            len_replay_buffer = len(replay_buffer_rewards)
            minibatches = 20*int(len_replay_buffer / self.batch)

            tensor_replay_reward = torch.tensor(self.norm(replay_buffer_rewards), dtype=torch.float).to(self.device,
                                                                                              non_blocking=True)
            tensor_replay_policy = torch.tensor(replay_buffer_policy, dtype=torch.float).to(self.device,
                                                                                             non_blocking=True)

            self.value_net.train()
            for iter in itertools.count():
                avg_loss = torch.tensor(0, dtype=torch.float)
                shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch),
                                                   replace=True)
                for i in range(minibatches):
                    samples = shuffle_indexes[i]
                    r = tensor_replay_reward[samples]
                    pi_explore = tensor_replay_policy[samples]
                    #r = torch.tensor(replay_buffer_rewards[samples], dtype=torch.float).to(self.device, non_blocking=True)
                    #pi_explore = torch.tensor(replay_buffer_policy[samples], dtype=torch.float).to(self.device, non_blocking=True)

                    self.optimizer_value.zero_grad()
                    q_value = -self.value_net(pi_explore).view(-1)
                    loss_q = self.q_loss(q_value, r).mean()
                    avg_loss += loss_q.item()
                    loss_q.backward()
                    self.optimizer_value.step()

                avg_loss /= minibatches
                if iter >= 1 or avg_loss < 0.1:
                    break


            self.value_net.eval()

            for _ in range(10):
                self.optimizer_beta.zero_grad()
                loss_beta = self.value_net(self.beta_net)
                loss_beta.backward()
                self.optimizer_beta.step()

            grads = self.beta_net.grad.detach().cpu().numpy().copy()
            self.save_checkpoint(self.checkpoint, {'n': self.frame})

            results['grads'].append(grads)
            results['reward_mean'].append(replay_buffer_rewards.mean())
            results['policies'].append(beta)
            q_value = self.value_net(torch.tensor(beta_explore, dtype=torch.float).to(self.device))
            results['q_value'].append(q_value.data.cpu().numpy())
            results['best_observed'].append(self.env.best_observed)
            results['ts'].append(self.env.t)
            results['q_loss'].append(avg_loss.numpy())

            yield results

            if len(results['best_observed']) > 700 and results['best_observed'][-1] == results['best_observed'][-700]:
                self.save_checkpoint(self.checkpoint, {'n': self.frame})
                print("VALUE IS NOT CHANGING - FRAME %d" % self.frame)
                break

            if results['ts'][-1]:
                self.save_checkpoint(self.checkpoint, {'n': self.frame})
                print("FINISHED SUCCESSFULLY - FRAME %d" % self.frame)
                break

            if self.frame >= self.budget:
                print("FAILED")
                break

    def norm(self, x):
        mean = x.mean()
        #std = x.std() + 1e-5
        #x = (x-mean)/std
        x = x - mean
        return x

    # def grad_explore(self, n_players):
    #     self.optimizer_beta.zero_grad()
    #     loss_beta = -self.value_net(self.beta_net)
    #     loss_beta.backward()
    #
    #     grads = self.beta_net.grad.detach().cpu().numpy().copy()
    #
    #     explore_factor = self.delta * grads + self.epsilon * np.random.randn(n_players, self.action_space)
    #     explore_factor *= 0.9 ** (2*np.array(range(n_players))).reshape(n_players,1)
    #     beta_explore = self.beta_net.detach().cpu().numpy() + explore_factor
    #
    #     beta_explore = np.clip(beta_explore, self.problem.lower_bounds, self.problem.upper_bounds)
    #
    #     return (beta_explore, grads)
    #
    # def uniform_explore(self, n_players):
    #
    #     explore_factor = self.epsilon * np.random.randn(n_players, self.action_space)
    #     explore_factor *= 0.9 ** (2 * np.array(range(n_players))).reshape(n_players, 1)
    #     beta_explore = self.beta_net.detach().cpu().numpy() + explore_factor
    #     pi_explore = np.exp(beta_explore) / np.sum(np.exp(beta_explore), axis=1).reshape(n_players, 1)
    #
    #     pi_explore = np.clip(pi_explore, self.problem.lower_bounds, self.problem.upper_bounds)
    #     return pi_explore
