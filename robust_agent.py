import torch
import torch.utils.data
import torch.utils.data.sampler
import torch.optim.lr_scheduler
import numpy as np
from tqdm import tqdm
import torch.autograd as autograd
from model_ddpg import RobustNormalizer
import itertools
from agent import Agent


class RobustAgent(Agent):

    def __init__(self, exp_name, env, checkpoint):
        super(RobustAgent, self).__init__(exp_name, env, checkpoint)
        reward_str = "Robust"
        print("Learning POLICY method using {} with RobustAgent".format(reward_str))
        
        assert self.batch == self.n_explore, 'n_explore diff from batch'
        
        self.mean = None
        self.std = None
        self.max = None
        self.min = None

        self.r_norm = RobustNormalizer()

    def print_robust_norm_params(self):
        if (self.frame % (self.printing_interval*self.n_explore)) == 0:
            print("\n\nframe {} -- r_norm: mu {} sigma {}".format(self.frame, self.r_norm.mu, self.r_norm.sigma))

    def update_replay_buffer(self):
        self.frame += self.warmup_explore
        explore_policies = self.exploration_rand(self.warmup_explore)
        self.step_policy(explore_policies)
        rewards = self.env.reward
        self.results['explore_policies'].append(explore_policies)
        self.results['rewards'].append(rewards)
        self.results_pi_update_with_explore(self.warmup_explore)
        self.tensor_replay_reward = rewards
        self.tensor_replay_policy = explore_policies

    def results_pi_update_with_explore(self, explore):
        pi = self.pi_net.pi.detach().cpu()
        pi_eval = self.step_policy(pi, to_env=False)
        best_observed = self.env.best_observed
        if not len(self.results['best_pi_evaluate']):
            best_pi_evaluate = pi_eval
        else:
            best_pi_evaluate = min(pi_eval, self.results['best_pi_evaluate'][-1])
        _, grad = self.get_grad()
        grad = grad.cpu().numpy().reshape(1, -1)
        dist_x = torch.norm(self.env.denormalize(self.pi_net().detach().cpu().numpy()) - self.best_op_x, 2)
        dist_f = pi_eval - self.best_op_f

        if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
            val = torch.norm(self.derivative_net(self.pi_net.pi.detach()).detach(), 2).detach().item()
            key = 'grad_norm'

        if self.algorithm_method in ['value', 'anchor']:
            val = self.r_norm.desquash(self.value_net(self.pi_net.pi.detach()).detach().cpu()).item()
            key = 'value'

        for _ in range (explore):
            self.results['best_observed'].append(best_observed)
            self.results['reward_pi_evaluate'].append(pi_eval)
            self.results['best_pi_evaluate'].append(best_pi_evaluate)
            self.results['policies'].append(pi)
            self.results['grad'].append(grad)
            self.results['dist_x'].append(dist_x)
            self.results['in_trust'].append(True)
            self.results['dist_f'].append(dist_f)
            self.results['mean_grad'].append(self.mean_grad)
            self.results['ts'].append(self.env.t)
            self.results['divergence'].append(self.divergence)
            self.results['r_norm_mean'].append(self.r_norm.mu)
            self.results['r_norm_sigma'].append(self.r_norm.sigma)
            self.results[key].append(val)

    def warmup(self):
        self.reset_net()
        self.r_norm.reset()
        self.update_replay_buffer()

        self.r_norm(self.tensor_replay_reward, training=True)
        self.mean = self.tensor_replay_reward.mean()
        self.std = self.tensor_replay_reward.std()
        self.max = self.tensor_replay_reward.max()
        self.min = self.tensor_replay_reward.min()

        self.value_optimize(100)

    def minimize(self):
        self.env.reset()
        self.warmup()
        #self.pi_net.eval()
        for _ in tqdm(itertools.count()):
            pi_explore, reward = self.exploration_step(self.n_explore)
            self.results['explore_policies'].append(pi_explore)
            self.results['rewards'].append(reward)

            self.value_optimize(self.value_iter)
            self.pi_optimize()

            self.save_checkpoint(self.checkpoint, {'n': self.frame})
            self.results_pi_update_with_explore(self.n_explore)
            # pi = self.pi_net.pi.detach().clone()
            # pi_eval = self.step_policy(pi, to_env=False)
            # self.results['best_observed'].append(self.env.best_observed)
            # self.results['reward_pi_evaluate'].append(pi_eval)
            # self.results['best_pi_evaluate'].append(min(self.results['reward_pi_evaluate']))
            # _, grad = self.get_grad()
            # self.results['grad'].append(grad.cpu().numpy().reshape(1, -1))
            # self.results['dist_x'].append(torch.norm(self.env.denormalize(self.pi_net().detach().cpu().numpy()) - self.best_op_x, 2))
            # self.results['dist_f'].append(pi_eval - self.best_op_f)
            #
            # self.results['policies'].append(pi)
            # if self.algorithm_method in ['first_order', 'second_order', 'anchor']:
            #     grad_norm = torch.norm(self.derivative_net(self.pi_net.pi.detach()).detach(), 2).detach().item()
            #     self.results['grad_norm'].append(grad_norm)
            # if self.algorithm_method in ['value', 'anchor']:
            #     value = self.r_norm.desquash(self.value_net(self.pi_net.pi.detach()).detach().cpu()).item()
            #     self.results['value'].append(np.array(value))
            #
            # self.results['ts'].append(self.env.t)
            # self.results['divergence'].append(self.divergence)

            yield self.results

            if self.results['ts'][-1]:
                self.save_checkpoint(self.checkpoint, {'n': self.frame})
                self.save_results(normalize_policy=True)
                print("FINISHED SUCCESSFULLY - FRAME %d" % self.frame)
                break

            if len(self.results['best_pi_evaluate']) > self.stop_con and self.results['best_pi_evaluate'][-1] == self.results['best_pi_evaluate'][-self.stop_con]:
                self.save_checkpoint(self.checkpoint, {'n': self.frame})
                self.save_results(normalize_policy=True)
                self.update_best_pi()
                self.epsilon *= 0.75
                self.update_pi_optimizer_lr()
                self.divergence += 1
                self.warmup()

            if self.divergence >= 10:
                print("DIVERGANCE - FAILED")
                break

            if self.frame >= self.budget:
                self.save_results(normalize_policy=True)
                print("FAILED")
                break

    def update_best_pi(self):
        rewards = np.hstack(self.results['rewards'])
        best_idx = rewards.argmin()
        pi = torch.cat(self.results['explore_policies'], dim=0)[best_idx]
        self.pi_net.pi_update(pi.to(self.device))

    def value_optimize(self, value_iter):

        self.tensor_replay_reward_norm = self.r_norm(self.tensor_replay_reward).to(self.device)
        self.tensor_replay_policy_norm = self.tensor_replay_policy.to(self.device)

        len_replay_buffer = len(self.tensor_replay_reward_norm)
        minibatches = len_replay_buffer // self.batch

        if self.algorithm_method == 'first_order':
            self.first_order_method_optimize_single_ref(len_replay_buffer, minibatches, value_iter)
            #self.first_order_method_optimize(len_replay_buffer, minibatches, value_iter)
        elif self.algorithm_method in ['value']:
            self.value_method_optimize(len_replay_buffer, minibatches, value_iter)
        elif self.algorithm_method == 'second_order':
            self.second_order_method_optimize(len_replay_buffer, minibatches, value_iter)
        elif self.algorithm_method == 'anchor':
            self.anchor_method_optimize(len_replay_buffer, minibatches, value_iter)
        else:
            raise NotImplementedError

    def anchor_method_optimize(self, len_replay_buffer, minibatches, value_iter):
        self.value_method_optimize(len_replay_buffer, minibatches)

        loss = 0
        self.derivative_net.train()
        self.value_net.eval()
        for it in itertools.count():
            shuffle_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=True)
            for i in range(minibatches):
                samples = shuffle_indexes[i]
                pi_1 = self.tensor_replay_policy_norm[samples]
                pi_tag_1 = self.derivative_net(pi_1)
                pi_2 = pi_1.detach() + torch.randn(self.batch, self.action_space).to(self.device, non_blocking=True)

                r_1 = self.tensor_replay_reward_norm[samples]
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

            if it >= value_iter:
                break

        loss /= value_iter
        self.results['derivative_loss'].append(loss)
        self.derivative_net.eval()


    def value_method_optimize(self, len_replay_buffer, minibatches, value_iter):
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
                q_value = self.value_net(pi_explore).view(-1)
                loss_q = self.q_loss(q_value, r).mean()
                loss += loss_q.detach().item()
                loss_q.backward()
                self.optimizer_value.step()

        loss /= value_iter
        self.results['value_loss'].append(loss)
        self.value_net.eval()

    def first_order_method_optimize(self, len_replay_buffer, minibatches, value_iter):

        loss = 0

        self.derivative_net.train()
        for _ in range(value_iter):
            anchor_indexes = np.random.choice(len_replay_buffer, (minibatches, self.batch), replace=False)
            batch_indexes = anchor_indexes // self.batch

            for i, anchor_index in enumerate(anchor_indexes):
                ref_index = torch.LongTensor(self.batch * batch_indexes[i][:, np.newaxis] + np.arange(self.batch)[np.newaxis, :])

                r_1 = self.tensor_replay_reward_norm[anchor_index].unsqueeze(1).repeat(1, self.batch)
                r_2 = self.tensor_replay_reward_norm[ref_index]
                pi_1 = self.tensor_replay_policy_norm[anchor_index]
                pi_2 = self.tensor_replay_policy_norm[ref_index]
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
                pi_tag_1 = self.derivative_net(pi_1)

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

        loss /= value_iter
        self.results['derivative_loss'].append(loss)
        self.derivative_net.eval()

    def step_policy(self, policy, to_env=True):
        policy = policy.cpu()
        policy = self.pi_net(policy)
        if to_env:
            self.env.step_policy(policy)
        else:
            return self.env.f(policy)

    def exploration_step(self, n_explore):
        self.frame += n_explore
        pi_explore = self.exploration(n_explore)
        self.step_policy(pi_explore)
        rewards = self.env.reward
        if self.best_explore_update:
            best_explore = rewards.argmin()
            self.pi_net.pi_update(pi_explore[best_explore].to(self.device))

        self.r_norm(rewards, training=True)

        self.tensor_replay_reward = torch.cat([self.tensor_replay_reward, rewards])[-self.replay_memory_size:]
        self.tensor_replay_policy = torch.cat([self.tensor_replay_policy, pi_explore])[-self.replay_memory_size:]

        #self.print_robust_norm_params()
        return pi_explore, rewards

    def get_evaluation_function(self, policy, target):
        policy = np.clip(policy, a_min=-1, a_max=1-1e-5)
        target = torch.FloatTensor(target)

        self.value_net.eval()
        batch = 1000
        value = []
        grads_norm = []
        for i in range(0, policy.shape[0], batch):
            from_index = i
            to_index = min(i + batch, policy.shape[0])
            policy_tensor = torch.FloatTensor(policy[from_index:to_index]).to(self.device)
            policy_tensor = autograd.Variable(policy_tensor, requires_grad=True)
            target_tensor = torch.FloatTensor(target[from_index:to_index]).to(self.device)
            q_value = self.value_net(policy_tensor, normalize=False).view(-1)
            value.append(q_value.detach().cpu().numpy())
            loss_q = self.q_loss(q_value, target_tensor).mean()
            grads = autograd.grad(outputs=loss_q, inputs=policy_tensor, grad_outputs=torch.cuda.FloatTensor(loss_q.size()).fill_(1.),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0].detach()
            grads_norm.append(torch.norm(torch.clamp(grads.view(-1, self.action_space), -1, 1), p=2, dim=1).cpu().numpy())

        value = np.hstack(value)
        grads_norm = np.hstack(grads_norm)
        pi = self.pi_net().detach().cpu().numpy()
        pi_value = self.value_net(self.pi_net.pi).detach().cpu().numpy()
        pi_with_grad = self.pi_net(self.pi_net.pi - self.pi_lr*self.get_grad()).detach().cpu().numpy()

        return value, pi, np.array(pi_value), pi_with_grad, grads_norm, self.r_norm(target).numpy()

    def get_grad_norm_evaluation_function(self, policy, f):
        policy = np.clip(policy, a_min=-1, a_max=1-1e-5)
        f = torch.FloatTensor(f)
        self.derivative_net.eval()
        policy_tensor = torch.FloatTensor(policy).to(self.device)
        policy_diff = policy_tensor[1:]-policy_tensor[:-1]
        policy_diff_norm = policy_diff / torch.norm(policy_diff, p=2, dim=1, keepdim=True)
        grad_direct = (policy_diff_norm * self.derivative_net(policy_tensor[:-1], normalize=False).detach()).sum(dim=1).cpu().numpy()
        pi = self.pi_net().detach().cpu().numpy()
        pi_grad = self.derivative_net(self.pi_net.pi).detach()
        pi_with_grad = pi - self.pi_lr*pi_grad.cpu().numpy()
        pi_grad_norm = torch.norm(pi_grad).cpu().numpy()
        return grad_direct, pi, pi_grad_norm, pi_with_grad, self.r_norm(f).numpy()
