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
from vae import VaeProblem, VAE
from environment import EnvCoco, EnvVae, EnvOneD
from collections import defaultdict

filter_mod = 15
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

        self.set_env()

    def set_env(self):
        if self.action_space == 784:
            self.env = EnvVae(self.problem)
        elif self.action_space == 1:
            self.env = EnvOneD(self.problem, True)
        else:
            self.env = EnvCoco(self.problem, True)


def main():

    set_seed(args.seed)
    username = pwd.getpwuid(os.geteuid()).pw_name
    algorithm = args.algorithm

    torch.set_num_threads(1000)
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
        divergence = run_exp(main_run.env, problem_index)
    else:
        res_dir = os.path.join('/data/', username, 'gan_rl', 'baseline', 'results', algorithm)
        if not os.path.exists(res_dir):
            try:
                os.makedirs(res_dir)
            except:
                pass

        for i in range(0, 360, filter_mod):
            main_run.reset(problem_index)
            divergence = run_exp(main_run.env, i)

            data['iter_index'].append(i)
            data['divergence'].append(divergence)
            data['index'].append(main_run.env.index)
            data['hit'].append(main_run.env.final_target_hit)
            data['id'].append(main_run.env.id)
            data['dimension'].append(main_run.env.dimension)
            data['best_observed'].append(main_run.env.best_observed_fvalue1)
            data['initial_solution'].append(main_run.env.initial_solution)
            data['upper_bound'].append(main_run.env.upper_bounds)
            data['lower_bound'].append(main_run.env.lower_bounds)
            data['number_of_evaluations'].append(main_run.env.evaluations)

            df = pd.DataFrame(data)
            fmin_file = os.path.join(res_dir, algorithm + '_' + str(args.action_space) + '.csv')
            df.to_csv(fmin_file)

    logger.info("End of simulation divergence = {}".format(divergence))

def run_exp(env, iter_index):
    with Experiment(logger.filename, env, iter_index) as exp:
        logger.info("BBO Session with VALUE net, it might take a while")
        divergence = exp.bbo()
    return divergence

if __name__ == '__main__':
    main()

