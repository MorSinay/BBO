import numpy as np
import torch

class Env(object):

    def __init__(self, problem_iter, need_norm=True, to_numpy=True):
        self.need_norm = need_norm
        self.problem_iter = problem_iter
        self.observed_list = []
        self.best_list = []
        self.pi_list = []
        self.to_numpy = to_numpy

    def get_observed_and_pi_list(self):
        return self.best_list, self.observed_list, self.pi_list

    def get_problem_dim(self):
        raise NotImplementedError

    def get_problem_index(self):
        raise NotImplementedError

    def get_problem_id(self):
        raise NotImplementedError

    def constrains(self):
         raise NotImplementedError

    def get_initial_solution(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step_policy(self, policy):
        raise NotImplementedError

    def f(self, policy):
        raise NotImplementedError

    def get_f0(self):
        raise NotImplementedError

    def denormalize(self):
        raise NotImplementedError

class EnvCoco(Env):

    def __init__(self, problem, problem_index, need_norm, to_numpy):
        super(EnvCoco, self).__init__(problem_index, need_norm, to_numpy)
        self.best_observed = None
        self.reward = None
        self.t = 0
        self.k = 0
        self.problem = problem
        self.output_size = self.problem.dimension

        self.reset()
        self.upper_bounds = self.problem.upper_bounds
        self.lower_bounds = self.problem.lower_bounds
        self.initial_solution = self.problem.initial_solution

        if self.need_norm:
            self.denormalize = self.with_denormalize
        else:
            self.denormalize = self.no_normalization

    def get_f0(self):
        return self.problem(self.initial_solution)

    def get_problem_dim(self):
        return self.problem.dimension

    def constrains(self):
         return self.lower_bounds, self.upper_bounds

    def get_initial_solution(self):
        return self.initial_solution

    def reset(self):
        self.best_observed = None
        self.reward = None
        self.k = 0
        self.t = 0

    def no_normalization(self, policy):
        policy = np.clip(policy, a_min=self.lower_bounds, a_max=self.upper_bounds)
        return policy

    def with_denormalize(self, policy):
        assert (np.max(policy) <= 1) or (np.min(policy) >= -1), "denormalized {}".format(policy)
        if len(policy.shape) == 2:
            assert (policy.shape[1] == self.output_size), "action error"
            upper = np.repeat(self.upper_bounds.reshape(1, -1), policy.shape[0], axis=0)
            lower = np.repeat(self.lower_bounds.reshape(1, -1), policy.shape[0], axis=0)
        else:
            upper = self.upper_bounds
            lower = self.lower_bounds

        policy = 0.5 * (policy + 1) * (upper - lower) + lower
        return policy

    def step_policy(self, policy):
        if self.to_numpy:
            policy = policy.cpu().numpy()
        policy = self.denormalize(policy)
        assert ((np.clip(policy, self.lower_bounds, self.upper_bounds) - policy).sum() < 0.000001), "clipping error {}".format(policy)
        self.reward = []
        if len(policy.shape) == 2:
            for i in range(policy.shape[0]):
                res = self.problem(policy[i])
                self.observed_list.append(res)
                self.best_list.append(self.problem.best_observed_fvalue1)
                self.reward.append(res)
                self.k += 1
        else:
            res = self.problem(policy)
            self.observed_list.append(res)
            self.best_list.append(self.problem.best_observed_fvalue1)
            self.reward.append(res)
            self.k += 1

        self.reward = torch.cuda.FloatTensor(self.reward)
        self.best_observed = self.problem.best_observed_fvalue1
        self.t = self.problem.final_target_hit

    def f(self, policy):
        if self.to_numpy:
            policy = policy.cpu().numpy()
        policy = self.denormalize(policy)
        res = self.problem(policy)
        self.observed_list.append(res)
        self.best_list.append(self.problem.best_observed_fvalue1)
        self.pi_list.append(policy)
        return res

    def get_problem_index(self):
        return self.problem.index

    def get_problem_id(self):
        return 'coco_' + str(self.problem.id)

class EnvVae(Env):

    def __init__(self, vae_problem, problem_index, to_numpy):
        super(EnvVae, self).__init__(problem_index, False, to_numpy)
        self.best_observed = None
        self.reward = None
        self.t = 0
        self.k = 0
        self.vae_problem = vae_problem
        self.output_size = self.vae_problem.dimension

        self.reset()
        self.upper_bounds = self.vae_problem.upper_bounds
        self.lower_bounds = self.vae_problem.lower_bounds
        self.initial_solution = self.vae_problem.initial_solution

    def get_problem_dim(self):
        return self.output_size

    def get_problem_index(self):
        return self.vae_problem.index

    def get_problem_id(self):
        return 'vae_' + str(self.vae_problem.id)

    def constrains(self):
         return self.lower_bounds, self.upper_bounds

    def get_initial_solution(self):
        return self.initial_solution

    def reset(self):
        self.best_observed = None
        self.reward = None
        self.k = 0
        self.t = 0

    def denormalize(self, policy):
        return policy

    def step_policy(self, policy):
        if self.to_numpy:
            policy = policy.cpu().numpy()
        assert ((np.clip(policy, self.lower_bounds, self.upper_bounds) - policy).sum() < 0.000001), "clipping error {}".format(policy)
        self.reward = []
        if len(policy.shape) == 2:
            for i in range(policy.shape[0]):
                res = self.vae_problem.func(policy[i])
                self.observed_list.append(res)
                self.best_list.append(self.problem.best_observed_fvalue1)
                self.reward.append(res)
                self.k += 1
        else:
            res = self.vae_problem.func(policy)
            self.observed_list.append(res)
            self.best_list.append(self.problem.best_observed_fvalue1)
            self.reward.append(res)
            self.k += 1

        self.reward = torch.cuda.FloatTensor(self.reward)
        self.best_observed = self.vae_problem.problem.best_observed_fvalue1
        self.t = self.vae_problem.problem.final_target_hit

    def f(self, policy):
        if self.to_numpy:
            policy = policy.cpu().numpy()
        res = self.vae_problem.func(policy)
        self.observed_list.append(res)
        self.best_list.append(self.problem.best_observed_fvalue1)
        self.pi_list.append(policy)
        return res

    def get_f0(self):
        return self.vae_problem.func(self.initial_solution)

class EnvOneD(Env):

    def __init__(self, problem, problem_index, need_norm, to_numpy):
        super(EnvOneD, self).__init__(problem_index, need_norm, to_numpy)
        self.best_observed = None
        self.reward = None
        self.t = 0
        self.k = 0
        self.problem = problem
        self.output_size = self.problem.dimension

        self.reset()
        self.upper_bounds = self.problem.upper_bounds[0]
        self.lower_bounds = self.problem.lower_bounds[0]
        self.initial_solution = np.array([self.problem.initial_solution[0]])

        if self.need_norm:
            self.denormalize = self.with_denormalize
        else:
            self.denormalize = self.no_normalization


    def get_f0(self):
        return self.problem(one_d_change_dim(self.initial_solution).flatten())

    def get_problem_dim(self):
        return self.output_size

    def get_problem_index(self):
        return self.problem.index

    def get_problem_id(self):
        return '1D_' + str(self.problem.id)

    def constrains(self):
         return self.lower_bounds, self.upper_bounds

    def get_initial_solution(self):
        return self.initial_solution

    def reset(self):
        self.best_observed = None
        self.reward = None
        self.k = 0
        self.t = 0

    def no_normalization(self, policy):
        policy = np.clip(policy, a_min=self.lower_bounds, a_max=self.upper_bounds)
        return policy

    def with_denormalize(self, policy):
        assert (np.max(policy) <= 1) or (np.min(policy) >= -1), "denormalized"
        if len(policy.shape) == 2:
            assert (policy.shape[1] == self.output_size), "action error, shape is {} and not {}".format(policy.shape[1], self.output_size)
            upper = np.repeat(self.upper_bounds.reshape(1, -1), policy.shape[0], axis=0)
            lower = np.repeat(self.lower_bounds.reshape(1, -1), policy.shape[0], axis=0)
        else:
            upper = self.upper_bounds
            lower = self.lower_bounds

        policy = 0.5 * (policy + 1) * (upper - lower) + lower
        return policy

    def step_policy(self, policy):
        if self.to_numpy:
            policy = policy.cpu().numpy()
        policy = self.denormalize(one_d_change_dim(policy))
        assert ((np.clip(policy, self.lower_bounds, self.upper_bounds) - policy).sum() < 0.000001), "clipping error {}".format(policy)
        self.reward = []
        if len(policy.shape) == 2:
            for i in range(policy.shape[0]):
                res = self.problem(policy[i])
                self.observed_list.append(res)
                self.best_list.append(self.problem.best_observed_fvalue1)
                self.reward.append(res)
                self.k += 1
        else:
            res = self.problem(policy)
            self.observed_list.append(res)
            self.best_list.append(self.problem.best_observed_fvalue1)
            self.reward.append(res)
            self.k += 1

        self.reward = torch.cuda.FloatTensor(self.reward)
        self.best_observed = self.problem.best_observed_fvalue1
        self.t = self.problem.final_target_hit

    def f(self, policy):
        if self.to_numpy:
            policy = policy.cpu().numpy()
        self.pi_list.append(policy)
        policy = self.denormalize(one_d_change_dim(policy)).flatten()
        res = self.problem(policy)
        self.observed_list.append(res)
        self.best_list.append(self.problem.best_observed_fvalue1)
        return res

def one_d_change_dim(policy):
    policy = policy.reshape(-1, 1)
    a = 1
    b = 0
    policy = np.hstack([policy, a * policy + b])
    policy = np.clip(policy, -1, 1)

    return policy
