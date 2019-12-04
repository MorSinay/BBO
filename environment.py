import numpy as np
from config import args

class Env(object):
    def __init__(self):
        self.normalize = args.normalize

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

    def normalize(self, policy):
        raise NotImplementedError

    def denormalize(self, policy):
        raise NotImplementedError


class EnvCoco(Env):

    def __init__(self, problem):
        super(EnvCoco, self).__init__()
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

    def normalize(self, policy):
        if self.normalize:
            policy = np.clip(policy, self.lower_bounds, self.upper_bounds)
            if len(policy.shape) == 2:
                assert (policy.shape[1] == self.output_size), "action error"
                upper = np.repeat(self.upper_bounds, policy.shape[0], axis=0)
                lower = np.repeat(self.lower_bounds, policy.shape[0], axis=0)
            else:
                upper = self.upper_bounds
                lower = self.lower_bounds

            policy = 2 * ((policy - lower) / upper - lower) - 1
        return policy

    def denormalize(self, policy):
        if self.normalize:
            assert (np.max(policy) <= 1) or (np.min(policy) >= -1), "denormalized"
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
        policy = self.denormalize(policy)
        assert ((np.clip(policy, self.lower_bounds, self.upper_bounds) - policy).sum() < 0.000001), "clipping error {}".format(policy)
        self.reward = []
        if len(policy.shape) == 2:
            for i in range(policy.shape[0]):
                self.reward.append(-self.problem(policy[i]))
                self.k += 1
        else:
            self.reward.append(-self.problem(policy))
            self.k += 1

        self.reward = np.array(self.reward)
        self.best_observed = self.problem.best_observed_fvalue1
        self.t = self.problem.final_target_hit

    def f(self, policy):
        policy = self.denormalize(policy)
        return self.problem(policy)

    def get_problem_index(self):
        return self.problem.index

    def get_problem_id(self):
        return self.problem.id

class EnvVae(Env):

    def __init__(self, vae_problem):
        super(EnvVae, self).__init__()
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
        return self.vae_problem.id

    def constrains(self):
         return self.lower_bounds, self.upper_bounds

    def get_initial_solution(self):
        return self.initial_solution

    def reset(self):
        self.best_observed = None
        self.reward = None
        self.k = 0
        self.t = 0

    def normalize(self, policy):
        return policy

    def denormalize(self, policy):
        return policy

    def step_policy(self, policy):
        assert ((np.clip(policy, self.lower_bounds, self.upper_bounds) - policy).sum() < 0.000001), "clipping error {}".format(policy)
        self.reward = []
        if len(policy.shape) == 2:
            for i in range(policy.shape[0]):
                self.reward.append(-self.vae_problem.func(policy[i]))
                self.k += 1
        else:
            self.reward.append(-self.vae_problem.func(policy))
            self.k += 1

        self.reward = np.array(self.reward)
        self.best_observed = self.vae_problem.problem.best_observed_fvalue1
        self.t = self.vae_problem.problem.final_target_hit

    def f(self, policy):
        return self.vae_problem.func(policy)
