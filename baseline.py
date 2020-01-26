import scipy.optimize
import pandas as pd
import numpy as np
import pwd
import os
from tqdm import tqdm
from collections import defaultdict
from environment2 import Env
from config import args, exp
import torch
import cma

epsilon = 1
filter_mod = 1
optimization_function = {'trust-ncg': scipy.optimize.minimize,
                         'trust-constr': scipy.optimize.minimize,
                         'trust-exact': scipy.optimize.minimize,
                         'trust-krylov': scipy.optimize.minimize,
                         'slsqp': scipy.optimize.fmin_slsqp,
                         'fmin': scipy.optimize.fmin,
                         'cobyla': scipy.optimize.fmin_cobyla,
                         'powell': scipy.optimize.fmin_powell,
                         'cg': scipy.optimize.fmin_cg,
                         'bfgs': scipy.optimize.fmin_bfgs,
                         'cma': cma.fmin2
}


def run_problem(alg, fmin, problem, x0, budget):

    x, best_val, eval_num = None, None, None

    try:
        if alg == 'slsqp':
            x, best_val, _, _, _ = fmin(problem, x0, iter=budget, full_output=True, iprint=-1)

        elif alg == 'fmin':
            x, best_val, _, eval_num, _ = fmin(problem, x0, maxfun=budget, disp=False, full_output=True)

        elif alg == 'cma':
            x, _ = fmin(problem, x0, 2, {'maxfevals': budget, 'verbose': -9}, restarts=0)

        elif alg == 'cobyla':
            x = fmin(problem, x0, cons=lambda x: None, maxfun=budget, disp=0, rhoend=1e-9)

        elif alg == 'powell':
            x, best_val, _, _, eval_num, _ = fmin(problem, x0, maxiter=budget, full_output=1)

        elif alg == 'cg':
            x, best_val, eval_num, _, _ = fmin(problem, x0, maxiter=budget, full_output=1)

        elif alg == 'bfgs':
            x, best_val, _, _, eval_num, _, _ = fmin(problem, x0, maxiter=budget, full_output=1)

        elif alg == 'trust-ncg':
            _ = fmin(problem, x0, args=(), method='trust-ncg', options={'maxiter': budget, 'disp': False})

        elif alg == 'trust-constr':
            _ = fmin(problem, x0, args=(), method='trust-constr', options={'maxiter': budget, 'disp': False})

        elif alg == 'trust-exact':
            _ = fmin(problem, x0, args=(), method='trust-exact', options={'maxiter': budget, 'disp': False})

        elif alg == 'trust-krylov':
            _ = fmin(problem, x0, args=(), method='trust-krylov', options={'maxiter': budget, 'disp': False})

        else:
            raise NotImplementedError

    except RuntimeError:
        pass

    return x, best_val, eval_num


class Baseline(object):

    def __init__(self):

        self.problem_index = args.problem_index
        self.alg = args.algorithm
        self.budget = args.budget

        self.env = Env(self.problem_index, display=True)
        self.results = defaultdict(lambda: defaultdict(list))

    def func(self, x):

        x = torch.cuda.FloatTensor(x).unsqueeze(0)

        bbo_results = self.env.step(x)

        r = float(bbo_results.reward)

        return r

    def learn(self):

        x0 = self.env.get_initial_solution().cpu().numpy()
        fmin = optimization_function[self.alg]

        x, best_val, eval_num = run_problem(self.alg, fmin, self.func, x0, self.budget)

        if x is not None:
            bbo_results = self.env.evaluate(torch.cuda.FloatTensor(x))
            reward, image, budget = bbo_results.reward, bbo_results.image, bbo_results.budget

            self.results['scalar']['reward'].append(reward)
            self.results['scalar']['budget'].append(budget)
            self.results['image']['current_image'] = image.detach().data.cpu().numpy()

        self.results['image']['target'] = self.env.image_target.data.cpu().numpy()
        self.results['text']['attributes'] = self.env.attributes_target_text

        self.results['scalar']['best_reward'].append(float(self.env.best_reward))
        self.results['image']['best_image'] = self.env.best_image.detach().data.cpu().numpy()

        self.results['aux']['attributes'] = self.env.attributes_target.squeeze(0).cpu().numpy()
        self.results['aux']['problem'] = int(self.env.problem)
        self.results['aux']['landmarks'] = self.env.landmark_target.data.cpu().numpy()

        return self.results



