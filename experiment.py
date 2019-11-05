import time
import os
import sys
import numpy as np
from tensorboardX import SummaryWriter

from config import consts, args, DirsAndLocksSingleton
from single_agent import BBOAgent

from logger import logger
from distutils.dir_util import copy_tree

import scipy.optimize  # to define the solver to be benchmarked

class Experiment(object):

    def __init__(self, logger_file, problem, suite_name):

        # parameters
        self.suite_name = suite_name
        self.action_space = args.action_space
        dirs = os.listdir(consts.outdir)

        self.load_model = args.load_last_model
        self.load_last = args.load_last_model
        self.resume = args.resume
        self.problem = problem
        self.problem_index = self.problem.index

        temp_name = "%s_%s_%s_bbo_%s" % (args.game, args.algorithm, args.identifier, args.problem_index)
        self.exp_name = ""
        if self.load_model:
            if self.resume >= 0:
                for d in dirs:
                    if "%s_%04d_" % (temp_name, self.resume) in d:
                        self.exp_name = d
                        self.exp_num = self.resume
                        break
            elif self.resume == -1:

                ds = [d for d in dirs if temp_name in d]
                ns = np.array([int(d.split("_")[-3]) for d in ds])
                self.exp_name = ds[np.argmax(ns)]
            else:
                raise Exception("Non-existing experiment")

        if not self.exp_name:
            # count similar experiments
            n = max([-1] + [int(d.split("_")[-3]) for d in dirs if temp_name in d]) + 1
            self.exp_name = "%s_%04d_%s" % (temp_name, n, consts.exptime)
            self.load_model = False
            self.exp_num = n

        # init experiment parameters
        self.dirs_locks = DirsAndLocksSingleton(self.exp_name)

        self.root = self.dirs_locks.root

        # set dirs
        self.tensorboard_dir = os.path.join(self.dirs_locks.tensorboard_dir, problem.id)
        self.checkpoints_dir = self.dirs_locks.checkpoints_dir
        self.results_dir = self.dirs_locks.results_dir
        self.code_dir = self.dirs_locks.code_dir
        self.analysis_dir = self.dirs_locks.analysis_dir
        self.checkpoint = self.dirs_locks.checkpoint

        if self.load_model:
            logger.info("Resuming existing experiment")
            with open(os.path.join(self.root, "logger"), "a") as fo:
                fo.write("%s resume\n" % logger_file)
        else:
            logger.info("Creating new experiment")
            # copy code to dir
            copy_tree(os.path.abspath("."), self.code_dir)

            # write args to file
            filename = os.path.join(self.root, "args.txt")
            with open(filename, 'w') as fp:
                fp.write('\n'.join(sys.argv[1:]))

            with open(os.path.join(self.root, "logger"), "a") as fo:
                fo.write("%s\n" % logger_file)

        # initialize tensorboard writer
        if args.tensorboard:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir, comment=args.identifier)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if args.tensorboard:
            self.writer.export_scalars_to_json(os.path.join(self.tensorboard_dir, "all_scalars.json"))
            self.writer.close()

    def print_statistics(self, n, beta, beta_explore, value, reward, best_observe, beta_evaluate, grads, finish, loss, r_mean):
        logger.info("----- INDEX -----: %d\t|----- DIM -----: %d" % (self.problem_index, self.action_space))
        logger.info("----- N -----: %d\t---- MEAN ----: %.2f|\t----- DONE -----: %d" % (n, r_mean, finish))
        logger.info("Actions statistics: |\t value = %f \t reward = %f \t q_loss = %f|" % (value, reward, loss))
        logger.info("Best observe      : |\t %f \t \tBeta_evaluate: = %f|" % (best_observe, beta_evaluate))

        beta_log         = "|\tbeta        \t"
        beta_explore_log = "|\tbeta_explore\t"
        grads_log        = "|\tgrads        \t"
        for i in range(args.action_space):
            beta_log += "|%.2f" % beta[i]
            beta_explore_log += "|%.2f" % beta_explore[i]
            grads_log += "|%.2f" % grads[i]
        beta_log += "|"
        beta_explore_log += "|"
        grads_log += "|"

        logger.info(beta_log)
        logger.info(beta_explore_log)
        logger.info(grads_log)

    def benchmarked_compare_scipy_fmin(self):
        budget = args.budget * self.problem.dimension
        propose_x0 = self.problem.initial_solution_proposal
        fmin = scipy.optimize.fmin
        output = fmin(self.problem, propose_x0(), maxfun=budget, disp=False, full_output=True)
        str_p = "FMIN - scipy.optimize.fmin\n------------------------------\n"
        str_p += "SUCCESS IN FINDING THE MINIMUM" if (
                    self.problem.final_target_hit == 1) else "FAILURE IN FINDING THE MINIMUM"
        str_p += "\nBest x value:"
        for i in range(self.problem.dimension):
            str_p += "|%.2f\t" % output[0][i]
        str_p += '\nFunction value: %.2f\nNumber of evaluations: %.2f' % (output[1], output[3])
        logger.info(str_p)

    def benchmarked_compare_scipy_fmin_slsqp(self):
        propose_x0 = self.problem.initial_solution_proposal

        fmin = scipy.optimize.fmin_slsqp
        output = fmin(self.problem, propose_x0(), iter=args.budget,# very approximate way to respect budget
                      full_output=True, iprint=-1)
        str_p = "FMIN - scipy.optimize.fmin_slsqp\n------------------------------\n"
        str_p += "SUCCESS IN FINDING THE MINIMUM" if (
                    self.problem.final_target_hit == 1) else "FAILURE IN FINDING THE MINIMUM"
        str_p += "\nBest x value:"
        for i in range(self.problem.dimension):
            str_p += "|%f\t" % output[0][i]
        str_p += '\nFunction value: %f\nNumber of evaluations: %.2f' % (output[1], self.problem.evaluations)
        logger.info(str_p)

    def bbo(self):

        agent = BBOAgent(self.exp_name, self.problem, checkpoint=self.checkpoint)

        n_explore = args.batch
        player = agent.find_min(n_explore)

        for n, bbo_results in (enumerate(player)):
            beta = bbo_results['policies'][-1]
            beta_explore = np.average(bbo_results['explore_policies'][-1], axis=0)
            #loss_value = bbo_results['loss_value'][-1] #TODO:
            value = -np.average(bbo_results['q_value'][-1])
            reward = np.average(bbo_results['rewards'][-1])
            best_observe = bbo_results['best_observed'][-1]
            grads = bbo_results['grads'][-1]
            beta_evaluate = self.problem(beta)
            finish = bbo_results['ts'][-1]
            loss = bbo_results['q_loss'][-1]
            r_mean = bbo_results['reward_mean'][-1]

            if not n % 20:
                self.print_statistics(n, beta, beta_explore, value, reward, best_observe, beta_evaluate, grads, finish, loss, r_mean)

            # log to tensorboard
            if args.tensorboard:

                self.writer.add_scalar('evaluation/t', finish, n)
                self.writer.add_scalars('evaluation/value_reward', {'value': value,'reward': reward}, n)
                self.writer.add_scalars('evaluation/beta_evaluate_observe', {'evaluate': beta_evaluate, 'best': best_observe}, n)


                for i in range(len(beta)):
                    self.writer.add_scalars('evaluation/beta_' + str(i), {'beta': beta[i], 'explore': beta_explore[i]}, n)
                    self.writer.add_scalar('evaluation/grad_' + str(i), grads[i], n)

                if hasattr(agent, "beta_net"):
                    self.writer.add_histogram("evaluation/beta_net", agent.beta_net.clone().cpu().data.numpy(), n, 'fd')
                if hasattr(agent, "value_net"):
                    for name, param in agent.value_net.named_parameters():
                        self.writer.add_histogram("evaluation/value_net/%s" % name, param.clone().cpu().data.numpy(), n, 'fd')

        print("End evaluation")

