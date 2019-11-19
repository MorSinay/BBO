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

    base_dir = os.path.join('/data/', username, 'gan_rl', 'baseline')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    torch.set_num_threads(1000)
    print("Torch %d" % torch.get_num_threads())
    # print args of current run
    logger.info("Welcome to Gan simulation")
    logger.info(' ' * 26 + 'Simulation Hyperparameters')
    for k, v in vars(args).items():
        logger.info(' ' * 26 + k + ': ' + str(v))

    suite_name = "bbob"
    suite_filter_options = ("dimensions: " + str(args.action_space)  #"year:2019 " +  "instance_indices: 1-5 "
                        )
    suite = cocoex.Suite(suite_name, "", suite_filter_options)
    data = defaultdict(list)
    problem_index = args.problem_index
    is_grad = args.grad
    divergence = 0
    for i, problem in enumerate(suite):
        if problem_index == -1:
            with Experiment(logger.filename, EnvCoco(problem)) as exp:
                if is_grad:
                    logger.info("BBO Session with GRADS net, it might take a while")
                    divergence = exp.bbo_with_grads()
                else:
                    logger.info("BBO Session with VALUE net, it might take a while")
                    divergence = exp.bbo()

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
            if is_grad:
                fmin_file = os.path.join(base_dir, 'grad_' + str(args.action_space) + '.csv')
            else:
                fmin_file = os.path.join(base_dir, 'bbo_' + str(args.action_space) + '.csv')
            df.to_csv(fmin_file)

        elif problem_index == i:
            with Experiment(logger.filename, EnvCoco(problem)) as exp:
                if args.grad:
                    logger.info("BBO Session with GRADS net, it might take a while")
                    divergence = exp.bbo_with_grads()
                else:
                    logger.info("BBO Session with VALUE net, it might take a while")
                    divergence = exp.bbo()

    logger.info("End of simulation divergence = {}".format(divergence))


def vae_simulation():

    set_seed(args.seed)
    username = pwd.getpwuid(os.geteuid()).pw_name

    base_dir = os.path.join('/data/', username, 'gan_rl', 'baseline')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    torch.set_num_threads(1000)
    print("Torch %d" % torch.get_num_threads())
    # print args of current run
    logger.info("Welcome to Gan simulation")
    logger.info(' ' * 26 + 'Simulation Hyperparameters')
    for k, v in vars(args).items():
        logger.info(' ' * 26 + k + ': ' + str(v))

    data = defaultdict(list)
    problem_index = args.problem_index
    is_grad = args.grad
    divergence = 0
    for i in range(360):
        if problem_index == -1:
            vae_problem = VaeProblem(i)
            with Experiment(logger.filename, EnvVae(vae_problem)) as exp:
                if is_grad:
                    logger.info("BBO Session with GRADS net, it might take a while")
                    divergence = exp.bbo_with_grads()
                else:
                    logger.info("BBO Session with VALUE net, it might take a while")
                    divergence = exp.bbo()

            data['iter_index'].append(i)
            data['divergence'].append(divergence)
            data['index'].append(i)
            data['hit'].append(vae_problem.final_target_hit)
            data['id'].append('vae_' + str(i))
            data['dimension'].append(vae_problem.dimension)
            data['best_observed'].append(vae_problem.best_observed_fvalue1)
            data['initial_solution'].append(vae_problem.initial_solution)
            data['upper_bound'].append(vae_problem.upper_bounds)
            data['lower_bound'].append(vae_problem.lower_bounds)
            data['number_of_evaluations'].append(vae_problem.evaluations)

            df = pd.DataFrame(data)
            if is_grad:
                fmin_file = os.path.join(base_dir, 'grad_' + str(args.action_space) + '.csv')
            else:
                fmin_file = os.path.join(base_dir, 'bbo_' + str(args.action_space) + '.csv')
            df.to_csv(fmin_file)

        elif problem_index == i:
            vae_problem = VaeProblem(i)
            with Experiment(logger.filename, EnvVae(vae_problem)) as exp:
                if args.grad:
                    logger.info("BBO Session with GRADS net, it might take a while")
                    divergence = exp.bbo_with_grads()
                else:
                    logger.info("BBO Session with VALUE net, it might take a while")
                    divergence = exp.bbo()

    logger.info("End of simulation divergence = {}".format(divergence))


if __name__ == '__main__':
    if args.action_space == 784:
        vae_simulation()
    else:
        main()

