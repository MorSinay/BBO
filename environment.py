import numpy as np

class Env(object):

    def __init__(self, problem):
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

    def step_policy(self, policy):

        assert ((np.clip(policy, self.lower_bounds, self.upper_bounds) - policy).sum() < 0.000001), "clipping error {}".format(policy)
        self.reward = []
        if len(policy.shape) == 2:
            assert(policy.shape[1] == self.output_size), "action error"
            for i in range(policy.shape[0]):
                self.reward.append(-self.problem(policy[i]))
                self.k += 1
        else:
            self.reward.append(-self.problem(policy))
            self.k += 1

        self.reward = np.array(self.reward)
        self.best_observed = self.problem.best_observed_fvalue1
        self.t = self.problem.final_target_hit
