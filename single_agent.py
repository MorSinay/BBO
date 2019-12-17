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

import os
import copy

import itertools
mem_threshold = consts.mem_threshold

#TODO:
# vae change logvar to
# curl regularization
# ELAD - 1d visualizetion of bbo function - eval function and derivative using projection in a specific direction


class BBOAgent(object):

    def __init__(self, exp_name, env, checkpoint):

        reward_str = "BBO"
        print("Learning POLICY method using {} with BBOAgent".format(reward_str))

        self.env = env
        self.dirs_locks = DirsAndLocksSingleton(exp_name)
        self.action_space = args.action_space
        self.cuda_id = args.cuda_default
        self.batch = args.batch
        self.warmup_minibatch = 2
        self.replay_memory_size = self.batch*args.replay_memory_factor
        self.replay_memory_factor = args.replay_memory_factor
        self.problem_index = args.problem_index
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
        self.max = None
        self.min = None
        self.transformer = None
        self.alpha = args.alpha
        self.pi_lr = args.beta_lr
        self.epsilon = args.epsilon
        self.delta = self.pi_lr

        if args.explore == 'grad_rand':
            self.exploration = self.exploration_grad_rand
        elif args.explore == 'grad_direct':
            self.exploration = self.exploration_grad_direct
        elif args.explore == 'grad_prop':
            self.exploration = self.exploration_grad_direct_prop
        elif args.explore == 'rand':
            self.exploration = self.exploration_rand
        elif args.explore == 'rand_test':
            self.exploration = self.exploration_rand_test
        else:
            print("explore:"+args.explore)
            raise NotImplementedError

        if args.norm == 'mean':
            self.output_norm = self.mean_norm
            self.output_denorm = self.mean_denorm
        elif args.norm == 'mean_std':
            self.output_norm = self.mean_std_norm
            self.output_denorm = self.mean_std_denorm
        elif args.norm == 'min_max':
            self.output_norm = self.min_max_norm
            self.output_denorm = self.min_max_denorm
        elif args.norm == 'no_norm':
            self.output_norm = self.no_norm
            self.output_denorm = self.no_denorm
        elif args.norm == 'robust_scaler':
            self.output_norm = self.robust_scaler_norm
            self.output_denorm = self.robust_scaler_denorm
        else:
            print("norm:" + args.norm)
            raise NotImplementedError

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.init = torch.tensor(self.env.get_initial_solution(), dtype=torch.float).to(self.device)
        self.pi_net = PiNet(self.init, self.device, self.action_space)
        self.optimizer_pi = torch.optim.SGD([self.pi_net.pi], lr=self.pi_lr)
        self.pi_net.train()

        if self.algorithm_method in ['first_order', 'second_order']:
            self.value_iter = 100
            self.derivative_net = DerivativeNet()
            self.derivative_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_derivative = torch.optim.Adam(self.derivative_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.derivative_net.train()
        elif self.algorithm_method == 'value':
            self.value_iter = 20
            self.value_net = DuelNet()
            self.value_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.value_net.eval()
        elif self.algorithm_method == 'spline':
            self.value_iter = 20
            self.spline_net = SplineNet(self.device, output=1)
            self.spline_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            opt_sparse = torch.optim.SparseAdam(self.spline_net.embedding.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-04)
            opt_dense = torch.optim.Adam(self.spline_net.head.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-04)
            self.optimizer_spline = MultipleOptimizer(opt_sparse, opt_dense)
            self.spline_net.eval()
        elif self.algorithm_method == 'anchor':
            self.value_iter = 100
            self.derivative_net = DerivativeNet()
            self.derivative_net.to(self.device)
            self.value_net = DuelNet()
            self.value_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_derivative = torch.optim.Adam(self.derivative_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.value_net.train()
            self.derivative_net.train()

        else:
            raise NotImplementedError

        self.q_loss = nn.SmoothL1Loss(reduction='none')

    def update_pi_optimizer_lr(self):
        op_dict = self.optimizer_pi.state_dict()
        self.pi_lr *= 0.97
        op_dict['param_groups'][0]['lr'] = self.pi_lr
        self.optimizer_pi.load_state_dict(op_dict)

    def save_results(self):
        for k in self.results.keys():
            path = os.path.join(self.analysis_dir, k +'.npy')
            if k in ['explore_policies']:
                policy = self.pi_net(torch.cat(self.results[k], dim=0))
                assert ((len(policy.shape) == 2) and (policy.shape[1] == self.action_space)), "save_results"
                np.save(path, policy.cpu().numpy())
            elif k in ['policies']:
                policy = self.pi_net(torch.stack(self.results[k]))
                assert ((len(policy.shape) == 2) and (policy.shape[1] == self.action_space)), "save_results"
                np.save(path, policy.cpu().numpy())
            else:
                tmp = np.array(self.results[k]).flatten()
                if tmp is None:
                    assert False, "save_results"
                np.save(path, tmp)

        path = os.path.join(self.analysis_dir, 'f0.npy')
        np.save(path, self.env.get_f0())

        if self.action_space == 784:
            path = os.path.join(self.analysis_dir, 'reconstruction.png')
            save_image(self.pi_net.pi.cpu().view(1, 28, 28), path)

    def save_checkpoint(self, path, aux=None):
        if self.algorithm_method in ['first_order', 'second_order']:
            state = {'pi_net': self.pi_net,
                     'derivative_net': self.derivative_net.state_dict(),
                     'optimizer_derivative': self.optimizer_derivative.state_dict(),
                     'optimizer_pi': self.optimizer_pi.state_dict(),
                     'aux': aux}
        elif self.algorithm_method == 'value':
            state = {'pi_net': self.pi_net,
                     'value_net': self.value_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_pi': self.optimizer_pi.state_dict(),
                     'aux': aux}
        elif self.algorithm_method == 'anchor':
            state = {'pi_net': self.pi_net,
                     'derivative_net': self.derivative_net.state_dict(),
                     'optimizer_derivative': self.optimizer_derivative.state_dict(),
                     'value_net': self.value_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_pi': self.optimizer_pi.state_dict(),
                     'aux': aux}
        elif self.algorithm_method == 'spline':
            # TODO: implement
            return
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
            # TODO: implement
            return None
        else:
            raise NotImplementedError
        self.n_offset = state['aux']['n']

        return state['aux']

    def warmup(self):
        explore_policies = self.exploration_rand(self.warmup_minibatch * self.batch)
        self.step_policy(explore_policies)
        rewards = self.env.reward
        self.results['explore_policies'].append(explore_policies)
        self.results['rewards'].append(rewards)
        self.mean = rewards.mean()
        self.std = rewards.std()
        self.max = rewards.max()
        self.min = rewards.min()
        self.transformer = RobustScaler().fit(rewards.reshape(-1,1))
        self.value_optimize()


    def minimize(self, n_explore):
        self.env.reset()
        self.warmup()
        #self.pi_net.eval()
        for self.frame in tqdm(itertools.count()):
            pi_explore, reward = self.exploration_step(n_explore)
            self.results['explore_policies'].append(pi_explore)
            self.results['rewards'].append(reward)

            self.value_optimize()
            self.pi_optimize()

            self.save_checkpoint(self.checkpoint, {'n': self.frame})

            pi = self.pi_net.pi.detach().clone()
            pi_eval = self.f_policy(pi)
            self.results['best_observed'].append(self.env.best_observed)
            self.results['pi_evaluate'].append(pi_eval)

            if self.bandage:
                self.bandage_update()

            self.results['policies'].append(pi)
            if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
                grad_norm = torch.norm(self.derivative_net(self.pi_net.pi.detach()).detach(), 2).detach().item()
                self.results['grad_norm'].append(grad_norm)
            if self.algorithm_method in ['value', 'anchor']:
                value = self.value_net(self.pi_net.pi.detach()).detach().item()
                self.results['value'].append(self.output_denorm(np.array(value)))
            if self.algorithm_method in ['spline']:
                value = self.spline_net(self.pi_net.pi.detach()).detach().item()
                self.results['value'].append(self.output_denorm(np.array(value)))

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

    def update_best_pi(self):
        rewards = np.hstack(self.results['rewards'])
        best_idx = rewards.argmin()
        pi = torch.cat(self.results['explore_policies'], dim=0)[best_idx]
        self.pi_net.pi_update(pi.to(self.device))

    def bandage_update(self):

        replay_observed = np.hstack(self.results['rewards'])[-self.replay_memory_size:]
        replay_evaluate = np.array(self.results['pi_evaluate'])[-self.replay_memory_factor:]

        best_observed = self.results['best_observed'][-1]
        best_evaluate = np.min(self.results['pi_evaluate'])

        bst_bandage = best_observed < np.min(replay_observed)
        bte_bandage = best_evaluate < np.min(replay_evaluate)

        if bst_bandage or bte_bandage:
            self.update_best_pi()
            self.update_pi_optimizer_lr()
            self.divergence += 1

    def value_optimize(self):
        #self.pi_net.eval()
        replay_buffer_rewards = np.hstack(self.results['rewards'])[-self.replay_memory_size:]
        replay_buffer_policy = torch.cat(self.results['explore_policies'], dim=0)[-self.replay_memory_size:]

        self.tensor_replay_reward = torch.tensor(self.output_norm(replay_buffer_rewards), dtype=torch.float).to(self.device, non_blocking=True)
        self.tensor_replay_policy = replay_buffer_policy.to(self.device)

        len_replay_buffer = len(replay_buffer_rewards)
        minibatches = len_replay_buffer // self.batch

        if self.algorithm_method == 'first_order':
            self.first_order_method_optimize(len_replay_buffer, minibatches)
        elif self.algorithm_method in ['value', 'spline']:
            self.value_method_optimize(len_replay_buffer, minibatches)
        elif self.algorithm_method == 'second_order':
            self.second_order_method_optimize(len_replay_buffer, minibatches)
        elif self.algorithm_method == 'anchor':
            self.anchor_method_optimize(len_replay_buffer, minibatches)
        else:
            raise NotImplementedError

    def anchor_method_optimize(self, len_replay_buffer, minibatches):
        self.value_method_optimize(len_replay_buffer, minibatches)

        loss = 0
        #self.derivative_net.train()
        #self.value_net.eval()
        for it in itertools.count():
            shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=True)
            for i in range(minibatches):
                samples = shuffle_indexes[i]
                pi_1 = self.tensor_replay_policy[samples]
                pi_tag_1 = self.derivative_net(pi_1)
                pi_2 = pi_1.detach() + torch.randn(self.batch, self.action_space).to(self.device, non_blocking=True)

                r_1 = self.tensor_replay_reward[samples]
                r_2 = self.value_net(pi_2).squeeze(1).detach()

                if self.importance_sampling:
                    w = torch.clamp((1 / (torch.norm(pi_2 - pi_1, p=2, dim=1) + 1e-4)).flatten(), 0, 1)
                else:
                    w = 1

                value = ((pi_2 - pi_1) * pi_tag_1).sum(dim=1)
                target = (r_2 - r_1)

                self.optimizer_derivative.zero_grad()
                self.optimizer_pi.zero_grad()
                loss_q = (w * self.q_loss(value, target)).mean()
                loss += loss_q.detach().item()
                loss_q.backward()
                self.optimizer_derivative.step()

            if it >= self.value_iter:
                break

        loss /= self.value_iter
        self.results['derivative_loss'].append(loss)
        #self.derivative_net.eval()


    def value_method_optimize(self, len_replay_buffer, minibatches):
        if self.algorithm_method in ['spline']:
            optimizer = self.optimizer_spline
            net = self.spline_net
        elif self.algorithm_method in ['value']:
            optimizer = self.optimizer_value
            net = self.value_net
        else:
            raise NotImplementedError

        loss = 0
        net.train()
        for it in itertools.count():
            shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=True)
            for i in range(minibatches):
                samples = shuffle_indexes[i]
                r = self.tensor_replay_reward[samples]
                pi_explore = self.tensor_replay_policy[samples]

                optimizer.zero_grad()
                self.optimizer_pi.zero_grad()
                q_value = net(pi_explore).view(-1)
                loss_q = self.q_loss(q_value, r).mean()
                loss += loss_q.detach().item()
                loss_q.backward()
                optimizer.step()

            if it >= self.value_iter:
                break

        loss /= self.value_iter
        self.results['value_loss'].append(loss)
        net.eval()

    def first_order_method_optimize(self, len_replay_buffer, minibatches):

        loss = 0
        #self.derivative_net.train()
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
            self.optimizer_pi.zero_grad()
            loss_q = (w * self.q_loss(value, target)).mean()
            loss += loss_q.detach().item()
            loss_q.backward()
            self.optimizer_derivative.step()

            if it >= self.value_iter:
                break

        loss /= self.value_iter
        self.results['derivative_loss'].append(loss)
        #self.derivative_net.eval()


    def second_order_method_optimize(self, len_replay_buffer, minibatches):
        loss = 0
        mid_val = True

        #self.derivative_net.train()
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
            self.optimizer_pi.zero_grad()
            loss_q = (w * self.q_loss(value, target)).mean()
            loss += loss_q.detach().item()
            loss_q.backward()
            self.optimizer_derivative.step()

            if it >= self.value_iter:
                break

        loss /= self.value_iter
        self.results['derivative_loss'].append(loss)
        #self.derivative_net.eval()

    def get_n_grad_ahead(self, n):

        optimizer_state = copy.deepcopy(self.optimizer_pi.state_dict())
        pi_array = [self.pi_net.pi.detach().clone()]
        for _ in range(n-1):
            pi, _ = self.grad_step()
            pi_array.append(pi)

        self.optimizer_pi.load_state_dict(optimizer_state)
        self.pi_net.pi_update(pi_array[0].to(self.device))

        pi_array = torch.stack(pi_array)
        return pi_array

    def grad_step(self):
        #self.pi_net.train()
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
        #self.pi_net.eval()
        pi = self.pi_net.pi.detach().clone()
        return pi, self.f_policy(pi)

    def mean_std_norm(self, data):
        self.mean = self.mean*(1-self.alpha) + self.alpha*data.mean()
        self.std = self.std*(1-self.alpha) + self.alpha*data.std()
        return (data - self.mean) / (self.std + 1e-5)

    def mean_std_denorm(self, data):
        return (data * (self.std + 1e-5)) + self.mean

    def mean_norm(self, data):
        self.mean = self.mean*(1-self.alpha) + self.alpha*data.mean()
        return data - self.mean

    def mean_denorm(self, data):
        return data + self.mean

    def min_max_norm(self, data):
        self.max = self.max*(1-self.alpha) + self.alpha*data.max()
        self.min = self.min*(1-self.alpha) + self.alpha*data.min()
        return (data - self.min) / (self.max - self.min + 1e-5)

    def min_max_denorm(self, data):
        return (data * (self.max - self.min + 1e-5)) + self.min

    def no_norm(self, data):
        return data

    def no_denorm(self, data):
        return data

    def robust_scaler_norm(self, data):
        data = data.reshape(-1, 1)
        self.transformer = RobustScaler().fit(data)
        return self.transformer.transform(data).flatten()

    def robust_scaler_denorm(self, data):
        data = data.reshape(-1, 1)
        return self.transformer.inverse_transform(data).flatten()

    def exploration_rand(self, n_explore):
            pi = self.pi_net.pi.detach().clone().cpu()
            pi_explore = pi + self.epsilon * torch.randn(n_explore, self.action_space)
            return pi_explore

    def exploration_rand_test(self, n_explore):
            pi_explore = -2 * torch.rand(n_explore, self.action_space) + 1
            return pi_explore

    def exploration_grad_rand_old(self, n_explore):
        grads = self.get_grad().cpu()
        pi = self.pi_net.pi.detach().clone().cpu()
        explore_factor = self.delta * grads + self.epsilon * torch.randn(n_explore, self.action_space)
        explore_factor *= 0.9 ** (2 * torch.arange(n_explore, dtype=torch.float)).reshape(n_explore, 1)
        pi_explore = pi - explore_factor  # gradient decent
        return pi_explore

    def exploration_grad_direct(self, n_explore):
        pi_array = self.get_n_grad_ahead(self.grad_steps).reshape(self.grad_steps, self.action_space).cpu()
        n_explore = (n_explore // self.grad_steps)

        epsilon_array = self.epsilon ** (3 - 2 * torch.arange(self.grad_steps, dtype=torch.float) / (self.grad_steps - 1))
        epsilon_array = epsilon_array.unsqueeze(1) # .expand_dims(epsilon_array, axis=1)
        pi_explore = torch.cat([pi_array + epsilon_array * torch.randn(self.grad_steps, self.action_space) for _ in range(n_explore)], dim=0)
        return pi_explore

    def exploration_grad_direct_prop(self, n_explore):
        pi_array = self.get_n_grad_ahead(self.grad_steps + 1).cpu()
        n_explore = (n_explore // self.grad_steps)

        diff_pi = torch.norm(pi_array[1:] - pi_array[:-1], 2, dim=1)

        epsilon_array = diff_pi * 10 ** (torch.arange(self.grad_steps, dtype=torch.float) / (self.grad_steps - 1))
        epsilon_array = epsilon_array.unsqueeze(1)#np.expand_dims(epsilon_array, axis=1)
        pi_explore = torch.cat([pi_array[:-1] - epsilon_array * torch.randn(self.grad_steps, self.action_space) for _ in range(n_explore)], dim=0)

        return pi_explore

    def get_grad(self):

        #self.pi_net.train()
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
            self.spline_net.zero_grad()
            loss_pi = self.spline_net(self.pi_net.pi)
            loss_pi.backward()
        else:
            raise NotImplementedError

        if self.grad_clip != 0:
            nn.utils.clip_grad_norm_(self.pi_net.pi, self.grad_clip / self.pi_lr)

        #self.pi_net.eval()
        grad = self.pi_net.pi.grad.detach().clone()
        return grad

    def f_policy(self, policy):
        policy = self.pi_net(policy)
        policy = policy.data.cpu().numpy()
        assert (policy.max() <= 1), "policy.max() {}".format(policy.max())
        assert (policy.min() >= -1), "policy.min() {}".format(policy.min())
        return self.env.f(policy)

    def step_policy(self, policy):
        policy = self.pi_net(policy)
        policy = policy.data.cpu().numpy()
        assert (policy.max() <= 1), "policy.max() {}".format(policy.max())
        assert (policy.min() >= -1), "policy.min() {}".format(policy.min())
        self.env.step_policy(policy)

    def pi_optimize(self):
        pi = self.pi_net.pi.detach().clone()
        pi_list = [pi]
        value_list = [self.f_policy(pi)]

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

    def exploration_step(self, n_explore):
        pi_explore = self.exploration(n_explore)
        self.step_policy(pi_explore)
        rewards = self.env.reward
        if self.best_explore_update:
            best_explore = rewards.argmin()
            self.pi_net.pi_update(pi_explore[best_explore].to(self.device))

        return pi_explore, rewards

    def get_evaluation_function(self, policy):
        if self.algorithm_method in 'value':
            net = self.value_net
        elif self.algorithm_method in 'spline':
            net = self.spline_net
        else:
            raise NotImplementedError

        #self.value_net.eval()
        policy_tensor = torch.tensor(policy, dtype=torch.float).to(self.device)
        value = net(policy_tensor).view(-1).detach().cpu().numpy()
        pi = self.pi_net().detach().cpu().numpy()
        pi_value = net(self.pi_net.pi).detach().cpu().numpy()
        grad = self.get_grad().cpu().numpy()
        return self.output_denorm(value), pi, self.output_denorm(np.array(pi_value)), self.pi_lr*grad
