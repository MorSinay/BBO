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
from environment import EnvCoco, EnvVae
from collections import defaultdict

filter_mod = 15
def set_seed(seed):
    if seed > 0:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

    suite_name = "bbob"
    suite_filter_options = ("dimensions: " + str(args.action_space))
    suite = cocoex.Suite(suite_name, "", suite_filter_options)

    if problem_index != -1:
        problem = suite.get_problem(problem_index)
        divergence = run_exp(EnvCoco(problem))
    else:
        res_dir = os.path.join('/data/', username, 'gan_rl', 'baseline', 'results', algorithm)
        if not os.path.exists(res_dir):
            try:
                os.makedirs(res_dir)
            except:
                pass

        for i in range(0, 360, filter_mod):
            problem = suite.get_problem(i)
            divergence = run_exp(EnvCoco(problem))

            data['iter_index'].append(i)
            data['divergence'].append(divergence)
            data['index'].append(problem.index)
            data['hit'].append(problem.final_target_hit)
            data['id'].append(problem.id)
            data['dimension'].append(problem.dimension)
            data['best_observed'].append(problem.best_observed_fvalue1)
            data['initial_solution'].append(problem.initial_solution)
            data['upper_bound'].append(problem.upper_bounds)
            data['lower_bound'].append(problem.lower_bounds)
            data['number_of_evaluations'].append(problem.evaluations)

            df = pd.DataFrame(data)
            fmin_file = os.path.join(res_dir, algorithm + '_' + str(args.action_space) + '.csv')
            df.to_csv(fmin_file)

    logger.info("End of simulation divergence = {}".format(divergence))


def run_exp(env):
    with Experiment(logger.filename, env) as exp:
        logger.info("BBO Session with VALUE net, it might take a while")
        divergence = exp.bbo()
    return divergence


def vae_simulation():

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

    if problem_index != -1:
        problem = VaeProblem(problem_index)
        divergence = run_exp(EnvVae(problem))
    else:
        res_dir = os.path.join('/data/', username, 'gan_rl', 'baseline', 'results', algorithm)
        if not os.path.exists(res_dir):
            try:
                os.makedirs(res_dir)
            except:
                pass

        for i in range(0, 360, filter_mod):
            problem = VaeProblem(problem_index)
            divergence = run_exp(EnvVae(problem))

            data['iter_index'].append(i)
            data['divergence'].append(divergence)
            data['index'].append(i)
            data['hit'].append(problem.final_target_hit)
            data['id'].append('vae_' + problem.id)
            data['dimension'].append(problem.dimension)
            data['best_observed'].append(problem.best_observed_fvalue1)
            data['initial_solution'].append(problem.initial_solution)
            data['upper_bound'].append(problem.upper_bounds)
            data['lower_bound'].append(problem.lower_bounds)
            data['number_of_evaluations'].append(problem.evaluations)

            df = pd.DataFrame(data)
            fmin_file = os.path.join(res_dir, algorithm + '_' + str(args.action_space) + '.csv')
            df.to_csv(fmin_file)

    logger.info("End of simulation divergence = {}".format(divergence))


if __name__ == '__main__':
    if args.action_space == 784:
        vae_simulation()
    else:
        main()

