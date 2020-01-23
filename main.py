from config import consts, args
from logger import logger
from experiment import Experiment
import torch
import cocoex
import pandas as pd
import os
import pwd
import random
import numpy as np
from gauss_uniform_vae import VaeProblem, VAE
from environment import EnvCoco, EnvVae, EnvOneD
from collections import defaultdict
import sys
import traceback

filter_mod = args.filter
problems_to_run = range(args.start, 360, filter_mod)
#problems_to_run = [15, 30, 45, 105, 120, 135, 150, 180, 210]
#problems_to_run = [105, 120, 135]
def set_seed(seed):
    if seed > 0:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class MainRun(object):

    def __init__(self):
        self.action_space = args.action_space
        self.problem = None
        if self.action_space != 784:
            suite_name = "bbob"
            suite_filter_options = ("dimensions: " + str(max(self.action_space, 2)))
            self.suite = cocoex.Suite(suite_name, "", suite_filter_options)

    def reset(self, problem_index):
        if self.action_space == 784:
            self.problem = VaeProblem(problem_index)
        else:
            self.suite.reset()
            self.problem = self.suite.get_problem(problem_index)

        self.set_env(problem_index)

    def set_env(self, problem_index):
        if self.action_space == 784:
            self.env = EnvVae(self.problem, problem_index, to_numpy=True)
        elif self.action_space == 1:
            self.env = EnvOneD(self.problem, problem_index, need_norm=True, to_numpy=True)
        else:
            self.env = EnvCoco(self.problem, problem_index, need_norm=True, to_numpy=True)

def main():

    set_seed(args.seed)
    username = pwd.getpwuid(os.geteuid()).pw_name
    algorithm = args.algorithm
    identifier = args.identifier
    run_id = algorithm + '_' + identifier

    torch.set_num_threads(100)
    print("Torch %d" % torch.get_num_threads())
    # print args of current run
    logger.info("Welcome to Gan simulation")
    logger.info(' ' * 26 + 'Simulation Hyperparameters')
    for k, v in vars(args).items():
        logger.info(' ' * 26 + k + ': ' + str(v))

    data = defaultdict(list)
    problem_index = args.problem_index
    divergence = 0
    main_run = MainRun()

    if problem_index != -1:
        main_run.reset(problem_index)
        divergence = run_exp(main_run.env)
    else:
        res_dir = os.path.join('/data/', username, 'gan_rl', 'baseline', 'results', run_id)
        if not os.path.exists(res_dir):
            try:
                os.makedirs(res_dir)
            except:
                pass

        for i in problems_to_run:
            main_run.reset(i)
            divergence = run_exp(main_run.env)

            data['iter_index'].append(i)
            data['divergence'].append(divergence)
            data['index'].append(main_run.env.problem.index)
            data['hit'].append(main_run.env.problem.final_target_hit)
            data['id'].append(main_run.env.get_problem_id())
            data['dimension'].append(main_run.env.problem.dimension)
            data['best_observed'].append(main_run.env.problem.best_observed_fvalue1)
            data['initial_solution'].append(main_run.env.initial_solution)
            data['upper_bound'].append(main_run.env.upper_bounds)
            data['lower_bound'].append(main_run.env.lower_bounds)
            data['number_of_evaluations'].append(main_run.env.problem.evaluations)

            df = pd.DataFrame(data)
            fmin_file = os.path.join(res_dir, run_id + '_' + str(args.action_space) + '.csv')
            df.to_csv(fmin_file)

    logger.info("End of simulation divergence = {}".format(divergence))

def run_exp(env):
    divergence = 0
    exp = Experiment(logger.filename, env)
    logger.info("BBO Session with NEURAL NET, it might take a while")
    try:
        divergence = exp.bbo()
    except Exception as e:
        logger.info(traceback.format_exc())
    return divergence

if __name__ == '__main__':
    main()

