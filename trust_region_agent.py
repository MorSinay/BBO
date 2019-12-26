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
from model_ddpg import DuelNet, DerivativeNet, PiClamp, SplineNet, MultipleOptimizer, RobustNormalizer, TrustRegion
import math
import os
import copy
from visualize_2d import get_best_solution
import itertools
mem_threshold = consts.mem_threshold

#TODO:
# vae change logvar to
# curl regularization
# ELAD - 1d visualizetion of bbo function - eval function and derivative using projection in a specific direction


class TrustRegionAgent(object):

    def __init__(self, exp_name, env, checkpoint):

        reward_str = "TrustRegion"
        print("Learning POLICY method using {} with TrustRegionAgent".format(reward_str))
        self.cuda_id = args.cuda_default
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.action_space = args.action_space
        self.env = env
        self.dirs_locks = DirsAndLocksSingleton(exp_name)

        self.best_op_x, self.best_op_f = get_best_solution(self.action_space, self.env.problem_iter)
        self.best_op_x = torch.FloatTensor(self.best_op_x).to(self.device)

        self.batch = args.batch
        self.n_explore = self.batch
        self.warmup_minibatch = args.warmup_minibatch
        self.replay_memory_size = self.batch*args.replay_memory_factor
        self.replay_memory_factor = args.replay_memory_factor
        self.problem_index = env.problem_iter
        self.value_lr = args.value_lr
        self.budget = args.budget
        self.checkpoint = checkpoint
        self.algorithm_method = args.algorithm
        self.grad_steps = args.grad_steps
        self.stop_con = args.stop_con
        self.grad_clip = args.grad_clip
        self.divergence = 0
        self.importance_sampling = args.importance_sampling
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
        self.tensor_replay_reward = None
        self.tensor_replay_policy = None
        self.mean = None
        self.std = None
        self.max = None
        self.min = None
        self.clip_up = np.inf
        self.clip_down = -np.inf
        self.alpha = args.alpha
        self.pi_lr = args.pi_lr
        self.epsilon = args.epsilon
        self.delta = self.pi_lr
        self.warmup_factor = args.warmup_factor

        if args.explore == 'grad_rand':
            self.exploration = self.exploration_grad_rand
        elif args.explore == 'grad_direct':
            self.exploration = self.exploration_grad_direct
        elif args.explore == 'rand':
            self.exploration = self.exploration_rand
        elif args.explore == 'rand_test':
            self.exploration = self.exploration_rand_test
        elif args.explore == 'cone':
            if self.action_space == 1:
                self.exploration = self.exploration_grad_direct
            else:
                self.exploration = self.cone_explore
        else:
            print("explore:"+args.explore)
            raise NotImplementedError

        self.init = torch.FloatTensor(self.env.get_initial_solution()).to(self.device)
        self.pi_net = PiClamp(self.init, self.device, self.action_space)
        self.optimizer_pi = torch.optim.SGD([self.pi_net.pi], lr=self.pi_lr)
        self.pi_net.eval()
        self.pi_trust_region = TrustRegion(self.pi_net)
        self.r_norm = RobustNormalizer()

        self.value_iter = args.learn_iteration
        if self.algorithm_method in ['first_order', 'second_order']:
            self.derivative_net = DerivativeNet(self.pi_net)
            self.derivative_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_derivative = torch.optim.Adam(self.derivative_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.derivative_net.eval()
            self.derivative_net_zero = copy.deepcopy(self.derivative_net.state_dict())
        elif self.algorithm_method == 'value':
            self.value_net = DuelNet(self.pi_net)
            self.value_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.value_net.eval()
            self.value_net_zero = copy.deepcopy(self.value_net.state_dict())
        elif self.algorithm_method == 'spline':
            self.spline_net = SplineNet(self.device, self.pi_net, output=1)
            self.spline_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            opt_sparse = torch.optim.SparseAdam(self.spline_net.embedding.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-04)
            opt_dense = torch.optim.Adam(self.spline_net.head.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-04)
            self.optimizer_spline = MultipleOptimizer(opt_sparse, opt_dense)
            self.spline_net.eval()
            self.spline_net_zero = copy.deepcopy(self.spline_net.state_dict())
        elif self.algorithm_method == 'anchor':
            self.derivative_net = DerivativeNet(self.pi_net)
            self.derivative_net.to(self.device)
            self.value_net = DuelNet(self.pi_net)
            self.value_net.to(self.device)
            # IT IS IMPORTANT TO ASSIGN MODEL TO CUDA/PARALLEL BEFORE DEFINING OPTIMIZER
            self.optimizer_derivative = torch.optim.Adam(self.derivative_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=self.value_lr, eps=1.5e-4, weight_decay=0)
            self.value_net.eval()
            self.derivative_net.eval()
            self.value_net_zero = copy.deepcopy(self.value_net.state_dict())
            self.derivative_net_zero = copy.deepcopy(self.derivative_net.state_dict())
        else:
            raise NotImplementedError


        self.q_loss = nn.SmoothL1Loss(reduction='none')

    def update_pi_optimizer_lr(self):
        op_dict = self.optimizer_pi.state_dict()
        self.pi_lr *= 0.85
        op_dict['param_groups'][0]['lr'] = self.pi_lr
        self.optimizer_pi.load_state_dict(op_dict)

    def save_results(self):
        for k in self.results.keys():
            path = os.path.join(self.analysis_dir, k +'.npy')
            if k in ['explore_policies']:
                policy = torch.cat(self.results[k], dim=0)
                assert ((len(policy.shape) == 2) and (policy.shape[1] == self.action_space)), "save_results"
                np.save(path, policy.cpu().numpy())
            elif k in ['policies']:
                policy = torch.stack(self.results[k])
                assert ((len(policy.shape) == 2) and (policy.shape[1] == self.action_space)), "save_results"
                np.save(path, policy.cpu().numpy())
            elif k in ['grad']:
                grad = np.vstack(self.results[k])
                assert ((len(grad.shape) == 2) and (grad.shape[1] == self.action_space)), "save_results"
                np.save(path, grad)
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
            state = {'pi_net': self.pi_net.pi.detach(),
                     'derivative_net': self.derivative_net.state_dict(),
                     'optimizer_derivative': self.optimizer_derivative.state_dict(),
                     'optimizer_pi': self.optimizer_pi.state_dict(),
                     'aux': aux}
        elif self.algorithm_method == 'value':
            state = {'pi_net': self.pi_net.pi.detach(),
                     'value_net': self.value_net.state_dict(),
                     'optimizer_value': self.optimizer_value.state_dict(),
                     'optimizer_pi': self.optimizer_pi.state_dict(),
                     'aux': aux}
        elif self.algorithm_method == 'anchor':
            state = {'pi_net': self.pi_net.pi.detach(),
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

    def print_robust_norm_params(self):
        if (self.frame % 50) == 0:
            print("\n\nframe {} -- r_norm: mu {} sigma {}".format(self.frame, self.r_norm.mu, self.r_norm.sigma))

    def warmup(self):
        if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
            self.derivative_net.load_state_dict(self.derivative_net_zero)
            self.optimizer_derivative.state = defaultdict(dict)
        if self.algorithm_method in ['value', 'anchor']:
            self.value_net.load_state_dict(self.value_net_zero)
            self.optimizer_value.state = defaultdict(dict)
        if self.algorithm_method in ['spline']:
            self.spline_net.load_state_dict(self.spline_net_zero)
            self.optimizer_spline.state = defaultdict(dict)

        self.r_norm.reset()

        explore_policies = self.exploration_rand(self.warmup_minibatch * self.n_explore)
        explore_policies = self.pi_net(explore_policies)
        self.step_policy(explore_policies)
        rewards = self.env.reward
        self.results['explore_policies'].append(self.pi_trust_region.inverse(explore_policies))
        self.results['rewards'].append(rewards)
        self.mean = rewards.mean()
        self.std = rewards.std()
        self.max = rewards.max()
        self.min = rewards.min()
        rewards_torch = torch.FloatTensor(rewards)
        self.r_norm(rewards_torch, training=True)
        self.tensor_replay_reward = torch.FloatTensor(rewards)
        self.tensor_replay_policy = torch.FloatTensor(explore_policies)
       # self.print_robust_norm_params()
        self.value_optimize(100)


    def minimize(self, n_explore):
        self.n_explore = n_explore
        self.env.reset()
        self.warmup()
        for self.frame in tqdm(itertools.count()):
            pi_explore, reward = self.exploration_step()
            self.results['explore_policies'].append(self.pi_trust_region.inverse(pi_explore))
            self.results['rewards'].append(reward)

            self.value_optimize(self.value_iter)
            self.pi_optimize()

            self.save_checkpoint(self.checkpoint, {'n': self.frame})

            pi = self.pi_net().detach().cpu()
            pi_eval = self.f_policy(pi)
            self.results['best_observed'].append(self.env.best_observed)
            self.results['pi_evaluate'].append(pi_eval)
            self.results['best_pi_evaluate'].append(min(self.results['pi_evaluate']))
            self.results['grad'].append(self.get_grad().cpu().numpy().reshape(1, -1))
            self.results['dist_x'].append(torch.norm(self.env.denormalize(self.pi_trust_region.inverse(pi).numpy()) - self.best_op_x, 2))
            self.results['dist_f'].append(pi_eval - self.best_op_f)

            self.results['policies'].append(self.pi_trust_region.inverse(pi))
            if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
                grad_norm = torch.norm(self.derivative_net(self.pi_net().detach(), normalize=False).detach(), 2).detach().item()
                self.results['grad_norm'].append(grad_norm)
            if self.algorithm_method in ['value', 'anchor']:
                value = self.r_norm.inverse(self.value_net(self.pi_net().detach(), normalize=False).detach().cpu()).item()
                self.results['value'].append(np.array(value))
            if self.algorithm_method in ['spline']:
                value = self.r_norm.inverse(self.spline_net(self.pi_net().detach(), normalize=False).detach().cpu()).item()
                self.results['value'].append(np.array(value))

            self.results['ts'].append(self.env.t)
            self.results['divergence'].append(self.divergence)

            yield self.results

            if self.results['ts'][-1]:
                self.save_checkpoint(self.checkpoint, {'n': self.frame})
                self.save_results()
                print("FINISHED SUCCESSFULLY - FRAME %d" % self.frame)
                break

            if len(self.results['best_pi_evaluate']) > self.stop_con and self.results['best_pi_evaluate'][-1] == self.results['best_pi_evaluate'][-self.stop_con]:
                self.save_checkpoint(self.checkpoint, {'n': self.frame})
                self.save_results()
                self.update_best_pi()
                # self.epsilon *= 0.75
                # self.update_pi_optimizer_lr()
                self.divergence += 1
                self.warmup()

            if self.divergence >= 10:
                print("DIVERGANCE - FAILED")
                break

            if self.frame >= self.budget:
                self.save_results()
                print("FAILED")
                break

    def update_best_pi(self):
        rewards = np.hstack(self.results['rewards'])
        best_idx = rewards.argmin()
        pi = torch.cat(self.results['explore_policies'], dim=0)[best_idx]

        self.pi_trust_region.squeeze(pi)
        self.pi_net.pi_update(self.pi_trust_region(pi).to(self.device))

    def value_optimize(self, value_iter):

        self.tensor_replay_reward_norm = self.r_norm(self.tensor_replay_reward).to(self.device)
        self.tensor_replay_policy_norm = self.tensor_replay_policy.to(self.device)

        len_replay_buffer = len(self.tensor_replay_reward_norm)
        minibatches = len_replay_buffer // self.batch

        if self.algorithm_method == 'first_order':
            self.first_order_method_optimize_single_ref(len_replay_buffer, minibatches, value_iter)
            #self.first_order_method_optimize(len_replay_buffer, minibatches, value_iter)
        elif self.algorithm_method in ['value', 'spline']:
            self.value_method_optimize(len_replay_buffer, minibatches, value_iter)
        elif self.algorithm_method == 'second_order':
            self.second_order_method_optimize(len_replay_buffer, minibatches, value_iter)
        elif self.algorithm_method == 'anchor':
            self.anchor_method_optimize(len_replay_buffer, minibatches, value_iter)
        else:
            raise NotImplementedError

    # def anchor_method_optimize(self, len_replay_buffer, minibatches, value_iter):
    #     self.value_method_optimize(len_replay_buffer, minibatches)
    #
    #     loss = 0
    #     self.derivative_net.train()
    #     self.value_net.eval()
    #     for it in itertools.count():
    #         shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=True)
    #         for i in range(minibatches):
    #             samples = shuffle_indexes[i]
    #             pi_1 = self.tensor_replay_policy_norm[samples]
    #             pi_tag_1 = self.derivative_net(pi_1)
    #             pi_2 = pi_1.detach() + torch.randn(self.batch, self.action_space).to(self.device, non_blocking=True)
    #
    #             r_1 = self.tensor_replay_reward_norm[samples]
    #             r_2 = self.value_net(pi_2).squeeze(1).detach()
    #
    #             if self.importance_sampling:
    #                 w = torch.clamp((1 / (torch.norm(pi_2 - pi_1, p=2, dim=1) + 1e-4)).flatten(), 0, 1)
    #             else:
    #                 w = 1
    #
    #             value = ((pi_2 - pi_1) * pi_tag_1).sum(dim=1)
    #             target = (r_2 - r_1)
    #
    #             self.optimizer_derivative.zero_grad()
    #             self.optimizer_pi.zero_grad()
    #             loss_q = (w * self.q_loss(value, target)).mean()
    #             loss += loss_q.detach().item()
    #             loss_q.backward()
    #             self.optimizer_derivative.step()
    #
    #         if it >= value_iter:
    #             break
    #
    #     loss /= value_iter
    #     self.results['derivative_loss'].append(loss)
    #     self.derivative_net.eval()


    def value_method_optimize(self, len_replay_buffer, minibatches, value_iter):
        if self.algorithm_method in ['spline']:
            optimizer = self.optimizer_spline
            net = self.spline_net
        elif self.algorithm_method in ['value', 'anchor']:
            optimizer = self.optimizer_value
            net = self.value_net
        else:
            raise NotImplementedError

        loss = 0
        net.train()
        for _ in range(value_iter):
            shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=False)
            for i in range(minibatches):
                samples = shuffle_indexes[i]
                r = self.tensor_replay_reward_norm[samples]
                pi_explore = self.tensor_replay_policy_norm[samples]

                optimizer.zero_grad()
                self.optimizer_pi.zero_grad()
                q_value = net(pi_explore, normalize=False).view(-1)
                loss_q = self.q_loss(q_value, r).mean()
                loss += loss_q.detach().item()
                loss_q.backward()
                optimizer.step()

        loss /= value_iter
        self.results['value_loss'].append(loss)
        net.eval()

    def first_order_method_optimize(self, len_replay_buffer, minibatches, value_iter):

        loss = 0

        self.derivative_net.train()
        for _ in range(value_iter):
            anchor_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=False)
            batch_indexes = anchor_indexes // self.n_explore

            for i, anchor_index in enumerate(anchor_indexes):
                ref_index = torch.LongTensor(self.batch * batch_indexes[i][:, np.newaxis] + np.arange(self.batch)[np.newaxis, :])

                r_1 = self.tensor_replay_reward_norm[anchor_index].unsqueeze(1).repeat(1, self.batch)
                r_2 = self.tensor_replay_reward_norm[ref_index]
                pi_1 = self.tensor_replay_policy_norm[anchor_index]
                pi_2 = self.tensor_replay_policy_norm[ref_index]
                pi_tag_1 = self.derivative_net(pi_1, normalize=False).unsqueeze(1).repeat(1, self.batch, 1)
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

        loss /= value_iter
        self.results['derivative_loss'].append(loss)
        self.derivative_net.eval()

    def first_order_method_optimize_single_ref(self, len_replay_buffer, minibatches, value_iter):

        loss = 0
        self.derivative_net.train()
        for _ in range(value_iter):
            anchor_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=False)
            ref_indexes = np.random.randint(0, self.batch, size=(minibatches, self.batch))
            batch_indexes = anchor_indexes // self.batch

            for i, anchor_index in enumerate(anchor_indexes):
                ref_index = torch.LongTensor(self.batch * batch_indexes[i] + ref_indexes[i])

                r_1 = self.tensor_replay_reward_norm[anchor_index]
                r_2 = self.tensor_replay_reward_norm[ref_index]
                pi_1 = self.tensor_replay_policy_norm[anchor_index]
                pi_2 = self.tensor_replay_policy_norm[ref_index]
                pi_tag_1 = self.derivative_net(pi_1, normalize=False)

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

        loss /= value_iter
        self.results['derivative_loss'].append(loss)
        self.derivative_net.eval()

    def first_order_method_optimize_temp(self, len_replay_buffer, minibatches, value_iter):

        loss = 0
        self.derivative_net.train()
        for _ in range(value_iter):
            shuffle_index = np.random.randint(0, self.batch, size=(minibatches,)) + np.arange(0, self.batch * minibatches, self.batch)
            r_1 = self.tensor_replay_reward_norm[shuffle_index].unsqueeze(1).repeat(1, self.batch)
            r_2 = self.tensor_replay_reward_norm.view(minibatches, self.batch)
            pi_1 = self.tensor_replay_policy_norm[shuffle_index]
            pi_2 = self.tensor_replay_policy_norm.view(minibatches, self.batch, -1)
            pi_tag_1 = self.derivative_net(pi_1, normalize=False).unsqueeze(1).repeat(1, self.batch, 1)
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

        loss /= value_iter
        self.results['derivative_loss'].append(loss)
        self.derivative_net.eval()

    def second_order_method_optimize(self, len_replay_buffer, minibatches, value_iter):
        loss = 0
        mid_val = True

        self.derivative_net.train()
        for _ in range(value_iter):

            shuffle_index = np.random.randint(0, self.batch, size=(minibatches,)) + np.arange(0, self.batch * minibatches, self.batch)
            r_1 = self.tensor_replay_reward_norm[shuffle_index].unsqueeze(1).repeat(1, self.batch)
            r_2 = self.tensor_replay_reward_norm.view(minibatches, self.batch)
            pi_1 = self.tensor_replay_policy_norm[shuffle_index].unsqueeze(1).repeat(1, self.batch, 1).reshape(-1,self.action_space)
            pi_2 = self.tensor_replay_policy_norm.view(minibatches * self.batch, -1)
            delta_pi = pi_2 - pi_1
            if mid_val:
                mid_pi = (pi_1 + pi_2) / 2
                mid_pi = autograd.Variable(mid_pi, requires_grad=True)
                pi_tag_mid = self.derivative_net(mid_pi, normalize=False)
                pi_tag_1 = self.derivative_net(pi_1, normalize=False)
                pi_tag_mid_dot_delta = (pi_tag_mid * delta_pi).sum(dim=1)
                gradients = autograd.grad(outputs=pi_tag_mid_dot_delta, inputs=mid_pi, grad_outputs=torch.cuda.FloatTensor(pi_tag_mid_dot_delta.size()).fill_(1.),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0].detach()
                second_order = 0.5 * (delta_pi * gradients.detach()).sum(dim=1)
            else:
                pi_1 = autograd.Variable(pi_1, requires_grad=True)
                pi_tag_1 = self.derivative_net(pi_1, normalize=False)
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

        loss /= value_iter
        self.results['derivative_loss'].append(loss)
        self.derivative_net.eval()

    def get_n_grad_ahead(self, n):

        optimizer_state = copy.deepcopy(self.optimizer_pi.state_dict())
        pi_array = [self.pi_net.pi.detach().clone()]
        for _ in range(n):
            pi, _ = self.grad_step()
            pi_array.append(pi)

        self.optimizer_pi.load_state_dict(optimizer_state)
        self.pi_net.pi_update(pi_array[0].to(self.device))

        pi_array = torch.stack(pi_array)
        return pi_array

    def grad_step(self):
        self.pi_net.train()
        self.optimizer_pi.zero_grad()
        if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
            self.optimizer_derivative.zero_grad()
            grad = self.derivative_net(self.pi_net(), normalize=False).detach().squeeze(0)
            self.pi_net.grad_update(grad.clone())
        elif self.algorithm_method == 'value':
            self.optimizer_value.zero_grad()
            loss_pi = self.value_net(self.pi_net(), normalize=False)
            loss_pi.backward()
        elif self.algorithm_method == 'spline':
            self.optimizer_spline.zero_grad()
            loss_pi = self.spline_net(self.pi_net(), normalize=False)
            loss_pi.backward()
        else:
            raise NotImplementedError

        if self.grad_clip != 0:
            nn.utils.clip_grad_norm_(self.pi_net.pi, self.grad_clip / self.pi_lr)

        self.optimizer_pi.step()
        self.pi_net.eval()
        pi = self.pi_net.pi.detach().clone()
        return pi, self.f_policy(pi.cpu())

    def exploration_rand(self, n_explore):
            pi = self.pi_net.pi.detach().clone().cpu()
            pi_explore = pi + self.warmup_factor*self.epsilon * torch.randn(n_explore, self.action_space)
            return pi_explore

    def exploration_rand_test(self, n_explore):
            pi_explore = -2 * torch.rand(n_explore, self.action_space) + 1
            return pi_explore

    def exploration_grad_rand(self, n_explore):
        grads = self.get_grad().cpu()
        pi = self.pi_net.pi.detach().clone().cpu()
        explore_factor = self.delta * grads + self.epsilon * torch.randn(n_explore, self.action_space)
        explore_factor *= 0.9 ** (2 * torch.arange(n_explore, dtype=torch.float)).reshape(n_explore, 1)
        pi_explore = pi - explore_factor  # gradient decent
        return pi_explore

    def exploration_grad_direct(self, n_explore):
        pi_array = self.get_n_grad_ahead(self.grad_steps).reshape(self.grad_steps+1, self.action_space).cpu()
        n_explore = (n_explore // self.grad_steps)

        epsilon_array = self.epsilon ** (3 - 2 * torch.arange(self.grad_steps+1, dtype=torch.float) / (self.grad_steps))
        epsilon_array = epsilon_array.unsqueeze(1) # .expand_dims(epsilon_array, axis=1)
        pi_explore = torch.cat([pi_array + epsilon_array * torch.randn(self.grad_steps+1, self.action_space) for _ in range(n_explore)], dim=0)
        return pi_explore

    def cone_explore(self, n_explore):
        alpha = math.pi / 4
        n = n_explore - 1
        pi = self.pi_net.pi.detach().cpu()
        d = self.get_grad().cpu()
        m = len(pi)
        pi = pi.unsqueeze(0)

        x = torch.FloatTensor(n, m).normal_()
        mag = torch.FloatTensor(n, 1).uniform_()

        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)
        d = d / (torch.norm(d) + 1e-8)

        cos = (x @ d).unsqueeze(1)

        dp = x - cos * d.unsqueeze(0)

        dp = dp / torch.norm(dp, dim=1, keepdim=True)

        acos = torch.acos(torch.clamp(torch.abs(cos), 0, 1-1e-8))

        new_cos = torch.cos(acos * alpha / (math.pi / 2))
        new_sin = torch.sin(acos * alpha / (math.pi / 2))

        cone = new_sin * dp + new_cos * d

        explore = pi - self.epsilon * mag * cone

        if np.isnan(explore).any():
            debug_path = os.path.join(consts.baseline_dir, 'cone_debug')
            if not os.path.exists(debug_path):
                os.makedirs(debug_path)
            np.save(os.path.join(debug_path, 'cos.npy'), cos)
            np.save(os.path.join(debug_path, 'dp.npy'), dp)
            np.save(os.path.join(debug_path, 'acos.npy'), acos)
            np.save(os.path.join(debug_path, 'new_cos.npy'), new_cos)
            np.save(os.path.join(debug_path, 'new_sin.npy'), new_sin)
            np.save(os.path.join(debug_path, 'x.npy'), x)
            np.save(os.path.join(debug_path, 'd.npy'), d)
            assert not np.isnan(explore).any(), "cone {} new_cos {}  new sin  {} acos {} dp {} cos {}".format(np.isnan(cone).any(),
                                                                                                              np.isnan(new_cos).any(),
                                                                                                              np.isnan(new_sin).any(),
                                                                                                              np.isnan(acos).any(),
                                                                                                              np.isnan(dp).any(),
                                                                                                              np.isnan(cos).any())

        return torch.cat([pi, explore])

    def get_grad(self):

        self.pi_net.train()
        self.optimizer_pi.zero_grad()
        if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
            self.optimizer_derivative.zero_grad()
            grad = self.derivative_net(self.pi_net(), normalize=False).detach().squeeze(0)
            self.pi_net.grad_update(grad.clone())
        elif self.algorithm_method == 'value':
            self.optimizer_value.zero_grad()
            loss_pi = self.value_net(self.pi_net(), normalize=False)
            loss_pi.backward()
        elif self.algorithm_method == 'spline':
            self.spline_net.zero_grad()
            loss_pi = self.spline_net(self.pi_net(), normalize=False)
            loss_pi.backward()
        else:
            raise NotImplementedError

        if self.grad_clip != 0:
            nn.utils.clip_grad_norm_(self.pi_net.pi, self.grad_clip / self.pi_lr)

        self.pi_net.eval()
        grad = self.pi_net.pi.grad.detach().clone()
        return grad

    def f_policy(self, policy):
        policy = self.pi_trust_region.inverse(policy)
        policy = policy.data.cpu().numpy()
        return self.env.f(policy)

    def step_policy(self, policy):
        policy = self.pi_trust_region.inverse(policy)
        policy = policy.data.cpu().numpy()
        self.env.step_policy(policy)

    def pi_optimize(self):
        pi = self.pi_net.pi.detach().clone()
        pi_list = [pi]
        value_list = [self.f_policy(pi.cpu())]

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

    def exploration_step(self):
        pi_explore = self.exploration(self.n_explore)
        pi_explore = self.pi_net(pi_explore)
        self.step_policy(pi_explore)
        rewards = self.env.reward
        if self.best_explore_update:
            best_explore = rewards.argmin()
            self.pi_net.pi_update(pi_explore[best_explore].to(self.device))

        rewards_torch = torch.FloatTensor(rewards)
        self.r_norm(rewards_torch, training=True)

        self.tensor_replay_reward = torch.cat([self.tensor_replay_reward, torch.FloatTensor(rewards)])
        self.tensor_replay_policy = torch.cat([self.tensor_replay_policy, torch.FloatTensor(pi_explore)])

        self.print_robust_norm_params()
        return pi_explore, rewards

    def get_evaluation_function(self, policy, target):

        upper = (self.pi_trust_region.mu + self.pi_trust_region.sigma).numpy()
        lower = (self.pi_trust_region.mu - self.pi_trust_region.sigma).numpy()
        for i in range(self.action_space):
            policy[policy[:, i] > upper[i]] = upper
            policy[policy[:, i] < lower[i]] = lower

        target = torch.FloatTensor(target)
        if self.algorithm_method in 'value':
            net = self.value_net
        elif self.algorithm_method in 'spline':
            net = self.spline_net
        else:
            raise NotImplementedError

        net.eval()
        batch = 1000
        value = []
        grads_norm = []
        for i in range(0, policy.shape[0], batch):
            from_index = i
            to_index = min(i + batch, policy.shape[0])
            policy_tensor = torch.FloatTensor(policy[from_index:to_index])
            policy_tensor = self.pi_trust_region(policy_tensor).to(self.device)
            policy_tensor = autograd.Variable(policy_tensor, requires_grad=True)
            target_tensor = torch.FloatTensor(target[from_index:to_index]).to(self.device)
            q_value = net(policy_tensor, normalize=False).view(-1)
            value.append(q_value.detach().cpu().numpy())
            loss_q = self.q_loss(q_value, target_tensor).mean()
            grads = autograd.grad(outputs=loss_q, inputs=policy_tensor, grad_outputs=torch.cuda.FloatTensor(loss_q.size()).fill_(1.),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0].detach()
            grads_norm.append(torch.norm(torch.clamp(grads.view(-1, self.action_space), -1, 1), p=2, dim=1).cpu().numpy())

        value = np.hstack(value)
        grads_norm = np.hstack(grads_norm)
        pi = self.pi_net().detach().cpu().numpy()
        pi_value = net(self.pi_net(), normalize=False).detach().cpu().numpy()
        pi_with_grad = pi - self.pi_lr*self.get_grad().cpu().numpy()

        return value, self.pi_trust_region.inverse(pi), np.array(pi_value), self.pi_trust_region.inverse(pi_with_grad), grads_norm, self.r_norm(target).numpy()

    def get_grad_norm_evaluation_function(self, policy, f):
        upper = (self.pi_trust_region.mu + self.pi_trust_region.sigma).numpy()
        lower = (self.pi_trust_region.mu - self.pi_trust_region.sigma).numpy()
        for i in range(self.action_space):
            policy[policy[:, i] > upper[i]] = upper
            policy[policy[:, i] < lower[i]] = lower

        f = torch.FloatTensor(f)
        self.derivative_net.eval()
        policy_tensor = torch.FloatTensor(policy)
        policy_tensor = self.pi_trust_region(policy_tensor).to(self.device)
        policy_diff = policy_tensor[1:]-policy_tensor[:-1]
        policy_diff_norm = policy_diff / (torch.norm(policy_diff, p=2, dim=1, keepdim=True) + 1e-5)
        grad_direct = (policy_diff_norm * self.derivative_net(policy_tensor[:-1], normalize=False).detach()).sum(dim=1).cpu().numpy()
        pi = self.pi_net().detach().cpu()
        pi_grad = self.derivative_net(self.pi_net(), normalize=False).detach()
        pi_with_grad = pi - self.pi_lr*pi_grad.cpu()
        pi_grad_norm = torch.norm(pi_grad).cpu()
        return grad_direct, self.pi_trust_region.inverse(pi).numpy(), pi_grad_norm, self.pi_trust_region.inverse(pi_with_grad).numpy(), self.r_norm(f).numpy()
