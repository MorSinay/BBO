import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
import torch.autograd as autograd
from model_ddpg import RobustNormalizer2, RobustNormalizer, NoRobustNormalizer, TrustRegion, NoTrustRegion

import itertools
from agent import Agent
import os
from config import args

class TrustRegionAgent(Agent):

    def __init__(self, exp_name, env, checkpoint):
        super(TrustRegionAgent, self).__init__(exp_name, env, checkpoint)
        reward_str = "TrustRegion"
        print("Learning POLICY method using {} with TrustRegionAgent".format(reward_str))

        if self.use_trust_region:
            self.pi_trust_region = TrustRegion(self.pi_net)
        else:
            self.pi_trust_region = NoTrustRegion(self.pi_net)

        if args.r_norm_alg == 'log':
            self.r_norm = RobustNormalizer2(lr=args.robust_scaler_lr)
        elif args.r_norm_alg == 'none':
            self.r_norm = NoRobustNormalizer()
        else:
            self.r_norm = RobustNormalizer(lr=args.robust_scaler_lr)

        if self.algorithm_method == 'EGL':
            self.value_optimize_method = self.EGL_method_optimize
        elif self.algorithm_method in ['IGL']:
            self.value_optimize_method = self.IGL_method_optimize
        else:
            raise NotImplementedError

        self.best_pi = self.pi_net.pi.detach().clone()
        self.best_pi_evaluate = self.step_policy(self.best_pi, to_env=False)
        self.best_reward = self.best_pi_evaluate*torch.cuda.FloatTensor(1)
        self.f0 = self.best_pi_evaluate
        self.trust_region_con = args.trust_region_con
        self.min_iter = args.min_iter
        self.no_change = 0
        self.pertub = args.pertub

    def update_replay_buffer(self):

        # self.tensor_replay_reward = torch.cuda.FloatTensor([])
        # self.tensor_replay_policy = torch.cuda.FloatTensor([])

        self.frame += self.warmup_minibatch*self.n_explore
        explore_policies_rand = self.ball_explore(self.warmup_minibatch*self.n_explore)

        self.step_policy(explore_policies_rand)
        rewards_rand = self.env.reward

        best_explore = rewards_rand.argmin()
        if self.best_reward > rewards_rand[best_explore]:
            self.best_pi = self.pi_trust_region.unconstrained_to_real(explore_policies_rand[best_explore].detach().clone())
            self.best_reward = rewards_rand[best_explore]

        self.results['explore_policies'].append(self.pi_trust_region.unconstrained_to_real(explore_policies_rand))
        self.results['rewards'].append(rewards_rand)
        self.r_norm(rewards_rand, training=True)
        self.results['norm_rewards'].append(self.r_norm(rewards_rand, training=False))

        self.tensor_replay_reward = torch.cat([self.tensor_replay_reward, rewards_rand])[-self.replay_memory_size:]
        self.tensor_replay_policy = torch.cat([self.tensor_replay_policy, explore_policies_rand])[-self.replay_memory_size:]

    def results_pi_update_with_explore(self):

        self.results['frame'] = self.frame
        self.results['best_observed'] = self.env.best_observed
        self.results['best_pi_evaluate'] = self.best_pi_evaluate
        self.results['best_reward'] = self.best_reward.cpu().numpy()
        _, grad = self.get_grad()
        grad = grad.cpu().numpy().reshape(1, -1)
        self.results['grad'] = grad

        if self.algorithm_method in ['EGL']:
            val = torch.norm(self.derivative_net(self.pi_net.pi.detach()).detach(), 2).detach().item()
            self.results['grad_norm'] = val
        if self.algorithm_method in ['IGL']:
            val = self.r_norm.desquash(self.value_net(self.pi_net.pi.detach()).detach()).cpu().item()
            self.results['IGL'] = val

        self.results['mean_grad'] = self.mean_grad.cpu().numpy()
        self.results['divergence'] = self.divergence
        self.results['r_norm_mean'] = self.r_norm.mu.detach().item()
        self.results['r_norm_sigma'] = self.r_norm.sigma.detach().item()
        self.results['min_trust_sigma'] = self.pi_trust_region.sigma.min().item()
        self.results['no_change'] = self.no_change
        self.results['epsilon'] = self.epsilon

        self.save_results()

    def save_results(self):
        for k in self.results.keys():
            path = os.path.join(self.analysis_dir, k + '.npy')
            data_np = np.array([])
            if os.path.exists(path):
                data_np = np.load(path, allow_pickle=True)

            if k in ['explore_policies']:
                policy = (torch.cat(self.results[k], dim=0)).cpu().numpy()
                if len(data_np):
                    policy = np.concatenate([data_np, policy])
                np.save(path, policy)
            elif k in ['policies']:
                policy = (torch.stack(self.results[k])).cpu().numpy()
                if len(data_np):
                    policy = np.concatenate([data_np, policy])
                np.save(path, policy)
            elif k in ['reward_pi_evaluate', 'frame_pi_evaluate']:
                rewards = self.results[k]
                if len(data_np):
                    rewards = np.concatenate([data_np, rewards])
                np.save(path, rewards)
            elif k in ['rewards', 'norm_rewards']:
                rewards = torch.cat(self.results[k]).cpu().numpy()
                if len(data_np):
                    rewards = np.hstack([data_np, rewards])
                np.save(path, rewards)
            elif k in ['grad']:
                grad = self.results[k]
                if len(data_np):
                    grad = np.concatenate([data_np, grad])
                np.save(path, grad)
            else:
                data = np.array([self.results[k]])
                if len(data_np):
                    data = np.hstack([data_np, data])
                np.save(path, data)

        best_list, observed_list, _ = self.env.get_observed_and_pi_list()
        np.save(os.path.join(self.analysis_dir, 'best_list_with_explore.npy'), np.array(best_list))
        np.save(os.path.join(self.analysis_dir, 'observed_list_with_explore.npy'), np.array(best_list))

        path = os.path.join(self.analysis_dir, 'f0.npy')
        np.save(path, self.f0)

    def warmup(self):
        self.mean_grad = None
        self.r_norm.reset()
        self.update_replay_buffer()
        self.value_optimize(self.value_iter)

    def save_and_print_results(self):
        self.save_checkpoint(self.checkpoint, {'n': self.frame})
        self.results_pi_update_with_explore()

    def minimize(self):
        counter = -1
        self.env.reset()
        self.reset_net()
        self.warmup()
        for i in tqdm(itertools.count()):
            counter += 1
            pi_explore, reward = self.exploration_step()
            self.results['explore_policies'].append(self.pi_trust_region.unconstrained_to_real(pi_explore))
            self.results['rewards'].append(reward)
            self.results['norm_rewards'].append(self.r_norm(reward, training=False))

            pi = self.pi_net.pi.detach()
            pi_eval = self.step_policy(pi, to_env=False)
            self.results['reward_pi_evaluate'].append(pi_eval)
            self.results['frame_pi_evaluate'].append(self.frame)
            real_pi = self.pi_trust_region.unconstrained_to_real(pi)
            self.results['policies'].append(real_pi)

            self.value_optimize(self.value_iter)
            self.pi_optimize()

            if pi_eval < self.best_pi_evaluate:
                self.no_change = 0
                self.best_pi_evaluate = pi_eval
            else:
                self.no_change += 1

            if pi_eval < self.best_reward:
                self.best_reward = torch.cuda.FloatTensor(pi_eval)
                self.best_pi = real_pi

            if self.env.t:
                self.save_and_print_results()
                yield self.results
                print("FINISHED SUCCESSFULLY - FRAME %d" % self.frame)
                break

            elif self.frame >= self.budget:
                self.save_and_print_results()
                yield self.results
                print("FAILED frame = {}".format(self.frame))
                break

            elif counter > self.min_iter and self.no_change > self.trust_region_con:
                counter = 0
                self.divergence += 1
                self.reset_net()
                self.update_best_pi()
                self.save_and_print_results()
                yield self.results
                self.reset_result()
                self.warmup()

            elif (i+1) % self.printing_interval == 0:
                self.save_and_print_results()
                yield self.results
                self.reset_result()

    def update_best_pi(self):
        pi = self.best_pi.detach().clone()
        real_replay = self.pi_trust_region.unconstrained_to_real(self.tensor_replay_policy)
        self.pi_trust_region.squeeze(pi)
        self.epsilon *= self.epsilon_factor
        self.epsilon = max(self.epsilon, 1e-4)
        self.pi_net.pi_update(self.pi_trust_region.real_to_unconstrained(pi))
        self.tensor_replay_policy = self.pi_trust_region.real_to_unconstrained(real_replay)

    def pi_optimize(self):

        _, grad = self.get_grad(grad_step=True)

        norm_factor = self.epsilon_factor**self.divergence

        grad_norm = torch.clamp(torch.norm(grad), max=20)/norm_factor

        if self.mean_grad is None:
            self.mean_grad = grad_norm
        else:
            self.mean_grad = (1 - self.alpha) * self.mean_grad + self.alpha * grad_norm

    def value_optimize(self, value_iter):

        self.tensor_replay_reward_norm = self.r_norm(self.tensor_replay_reward)
        self.tensor_replay_policy_norm = self.tensor_replay_policy

        len_replay_buffer = len(self.tensor_replay_reward_norm)
        self.batch = min(self.max_batch, len_replay_buffer)
        minibatches = len_replay_buffer // self.batch

        self.value_optimize_method(len_replay_buffer, minibatches, value_iter)

    def IGL_method_optimize(self, len_replay_buffer, minibatches, value_iter):
        loss = 0
        self.value_net.train()
        for _ in range(value_iter):
            shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=False)
            for i in range(minibatches):
                samples = shuffle_indexes[i]
                r = self.tensor_replay_reward_norm[samples]
                pi_explore = self.tensor_replay_policy_norm[samples]

                self.optimizer_value.zero_grad()
                self.optimizer_pi.zero_grad()
                q_value = self.value_net(pi_explore).flatten()
                if self.spline:
                    loss_q = self.q_loss(q_value, r).sum()
                else:
                    loss_q = self.q_loss(q_value, r).mean()
                loss += loss_q.detach().item()
                loss_q.backward()
                self.optimizer_value.step()

        loss /= value_iter
        self.results['value_loss'].append(loss)
        self.value_net.eval()

    def ball_perturb(self, pi, eps):

        n_explore = len(pi)

        x = torch.cuda.FloatTensor(n_explore, self.action_space).normal_()
        mag = torch.cuda.FloatTensor(n_explore, 1).uniform_()

        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)

        explore = pi + eps * mag * x

        return explore

    def EGL_method_optimize(self, len_replay_buffer, minibatches, value_iter):

        loss = 0
        self.derivative_net.train()
        for _ in range(value_iter):
            anchor_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=False)
            ref_indexes = np.random.randint(0, self.n_explore, size=(minibatches, self.batch))
            explore_indexes = anchor_indexes // self.n_explore

            for i, anchor_index in enumerate(anchor_indexes):
                ref_index = torch.LongTensor(self.n_explore * explore_indexes[i] + ref_indexes[i])

                r_1 = self.tensor_replay_reward_norm[anchor_index]
                r_2 = self.tensor_replay_reward_norm[ref_index]
                pi_1 = self.tensor_replay_policy_norm[anchor_index]
                pi_1_perturb = self.ball_perturb(pi_1, eps=self.epsilon*self.pertub)

                pi_2 = self.tensor_replay_policy_norm[ref_index]
                pi_tag_1 = self.derivative_net(pi_1_perturb)

                value = ((pi_2 - pi_1) * pi_tag_1).sum(dim=1)
                target = (r_2 - r_1)

                self.optimizer_derivative.zero_grad()
                self.optimizer_pi.zero_grad()
                if self.spline:
                    loss_q = self.q_loss(value, target).sum()
                else:
                    loss_q = self.q_loss(value, target).mean()

                loss += loss_q.detach().item()
                loss_q.backward()
                self.optimizer_derivative.step()

        loss /= value_iter
        self.results['derivative_loss'] = loss
        self.derivative_net.eval()

    def step_policy(self, policy, to_env=True):
        policy = self.pi_trust_region.unconstrained_to_real(policy)
        if to_env:
            self.env.step_policy(policy)
        else:
            return self.env.f(policy)

    def exploration_step(self):
        self.frame += self.n_explore
        pi_explore = self.exploration(self.n_explore)
        self.step_policy(pi_explore)
        rewards = self.env.reward

        best_explore = rewards.argmin()
        if self.best_explore_update:
            self.pi_net.pi_update(pi_explore[best_explore])

        if self.best_reward > rewards[best_explore]:
            self.best_pi = self.pi_trust_region.unconstrained_to_real(pi_explore[best_explore].detach().clone())
            self.best_reward = rewards[best_explore]

        self.r_norm(rewards, training=True)

        self.tensor_replay_reward = torch.cat([self.tensor_replay_reward, rewards])[-self.replay_memory_size:]
        self.tensor_replay_policy = torch.cat([self.tensor_replay_policy, pi_explore])[-self.replay_memory_size:]

        return pi_explore, rewards

    def get_evaluation_function(self, policy, target):
        upper = max((self.pi_trust_region.mu + self.pi_trust_region.sigma).cpu().numpy(), 1-1e-5)
        lower = min((self.pi_trust_region.mu - self.pi_trust_region.sigma).cpu().numpy(), -1)
        policy = np.clip(policy, a_min=lower, a_max=upper)

        target = torch.FloatTensor(target)

        self.value_net.eval()
        batch = 1024
        value = []
        grads_norm = []
        for i in range(0, policy.shape[0], batch):
            from_index = i
            to_index = min(i + batch, policy.shape[0])
            policy_tensor = torch.cuda.FloatTensor(policy[from_index:to_index])
            policy_tensor = self.pi_trust_region.real_to_unconstrained(policy_tensor)
            policy_tensor = autograd.Variable(policy_tensor, requires_grad=True)
            target_tensor = torch.cuda.FloatTensor(target[from_index:to_index])
            q_value = self.value_net(policy_tensor).view(-1)
            value.append(q_value.detach().cpu().numpy())

            if self.spline:
                loss_q = self.q_loss(q_value, target_tensor).sum()
            else:
                loss_q = self.q_loss(q_value, target_tensor).mean()

            grads = autograd.grad(outputs=loss_q, inputs=policy_tensor, grad_outputs=torch.cuda.FloatTensor(loss_q.size()).fill_(1.),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0].detach()
            grads_norm.append(torch.norm(torch.clamp(grads.view(-1, self.action_space), -1, 1), p=2, dim=1).cpu().numpy())

        value = np.hstack(value)
        grads_norm = np.hstack(grads_norm)
        pi = self.pi_net.pi.detach().cpu()
        pi_value = self.value_net(self.pi_net.pi).detach().cpu().numpy()
        pi_with_grad = pi - self.pi_lr*self.get_grad().cpu()

        return value, self.pi_trust_region.unconstrained_to_real(pi).cpu().numpy(), np.array(pi_value), self.pi_trust_region.unconstrained_to_real(pi_with_grad).cpu().numpy(), grads_norm, self.r_norm(target).cpu().numpy()

    def get_grad_norm_evaluation_function(self, policy, f):
        upper = max((self.pi_trust_region.mu + self.pi_trust_region.sigma).cpu().numpy(), 1 - 1e-5)
        lower = min((self.pi_trust_region.mu - self.pi_trust_region.sigma).cpu().numpy(), -1)
        policy = np.clip(policy, a_min=lower, a_max=upper)

        f = torch.FloatTensor(f)
        self.derivative_net.eval()
        policy_tensor = torch.cuda.FloatTensor(policy)
        policy_tensor = self.pi_trust_region.real_to_unconstrained(policy_tensor)
        policy_diff = policy_tensor[1:]-policy_tensor[:-1]
        policy_diff_norm = policy_diff / (torch.norm(policy_diff, p=2, dim=1, keepdim=True) + 1e-5)
        grad_direct = (policy_diff_norm * self.derivative_net(policy_tensor[:-1]).detach()).sum(dim=1).cpu().numpy()
        pi = self.pi_net.pi.detach().cpu()
        pi_grad = self.derivative_net(self.pi_net.pi).detach()
        pi_with_grad = pi - self.pi_lr*pi_grad.cpu()
        pi_grad_norm = torch.norm(pi_grad).cpu()
        return grad_direct, self.pi_trust_region.unconstrained_to_real(pi).cpu().numpy(), pi_grad_norm, self.pi_trust_region.unconstrained_to_real(pi_with_grad).cpu().numpy(), self.r_norm(f).cpu().numpy()

