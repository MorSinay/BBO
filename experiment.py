import time
import os
import sys
import numpy as np
import torch
from tensorboardX import SummaryWriter
from single_agent import BBOAgent
#from np_agent_temp import NPAgent
from config import consts, args, DirsAndLocksSingleton
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from environment import one_d_change_dim
from logger import logger
from distutils.dir_util import copy_tree
import pickle
import pandas as pd


import scipy.optimize  # to define the solver to be benchmarked

class Experiment(object):

    def __init__(self, logger_file, env, iter_index):

        # parameters
        self.action_space = args.action_space
        dirs = os.listdir(consts.outdir)

        self.load_model = args.load_last_model
        self.load_last = args.load_last_model
        self.resume = args.resume
        self.env = env
        self.problem_id = self.env.get_problem_id()
        self.algorithm = args.algorithm
        self.iter_index = iter_index
        # temp_name = "%s_%s_%s_bbo_%s" % (args.game, args.algorithm, args.identifier, str(args.action_space))
        # self.exp_name = ""
        # if self.load_model:
        #     if self.resume >= 0:
        #         for d in dirs:
        #             if "%s_%04d_" % (temp_name, self.resume) in d:
        #                 self.exp_name = d
        #                 self.exp_num = self.resume
        #                 break
        #     elif self.resume == -1:
        #
        #         ds = [d for d in dirs if temp_name in d]
        #         ns = np.array([int(d.split("_")[-3]) for d in ds])
        #         self.exp_name = ds[np.argmax(ns)]
        #     else:
        #         raise Exception("Non-existing experiment")
        #
        # if not self.exp_name:
        #     # count similar experiments
        #     n = max([-1] + [int(d.split("_")[-3]) for d in dirs if temp_name in d]) + 1
        #     self.exp_name = "%s_%04d_%s" % (temp_name, n, consts.exptime)
        #     self.load_model = False
        #     self.exp_num = n

        self.exp_name = "%s_%s_%s_%s" % (args.game, self.algorithm, args.identifier, str(args.action_space))
        # init experiment parameters
        self.dirs_locks = DirsAndLocksSingleton(self.exp_name)

        self.root = self.dirs_locks.root

        # set dirs
        self.tensorboard_dir = os.path.join(self.dirs_locks.tensorboard_dir, self.problem_id)
        if not os.path.exists(self.tensorboard_dir):
            try:
                os.makedirs(self.tensorboard_dir)
            except:
                pass

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

        self.agent = None
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if args.tensorboard:
            self.writer.export_scalars_to_json(os.path.join(self.tensorboard_dir, "all_scalars.json"))
            self.writer.close()

    def bbo(self):
        self.agent = BBOAgent(self.exp_name, self.env, checkpoint=self.checkpoint)
        n_explore = args.batch
        player = self.agent.minimize(n_explore)
        divergence = 0

        for n, bbo_results in (enumerate(player)):
            pi = bbo_results['policies'][-1].cpu().numpy()
            pi_explore = torch.mean(bbo_results['explore_policies'][-1], dim=0).cpu().numpy()
            pi_evaluate = bbo_results['pi_evaluate'][-1]

            avg_reward = np.average(bbo_results['rewards'][-1])
            best_observe = bbo_results['best_observed'][-1]
            divergence = bbo_results['divergence'][-1]

            if not n % 20:
                logger.info("-------------------- iteration: {} - Problem ID :{} --------------------".format(n, self.problem_id))
                logger.info("Problem iter index     :{}\t\t\tDim: {}\t\t\tDivergence: {}".format(self.iter_index, self.action_space, divergence))
                if self.algorithm in ['first_order', 'second_order']:
                    logger.info("Actions statistics: |\t grad norm = %.3f \t avg_reward = %.3f| \t derivative_loss =  %.3f" % (bbo_results['grad_norm'][-1], avg_reward, bbo_results['derivative_loss'][-1]))
                elif self.algorithm == ['value', 'spline']:
                    logger.info("Actions statistics: |\t value = %.3f \t avg_reward = %.3f \t value_loss =  %.3f|" % (bbo_results['value'][-1], avg_reward, bbo_results['value_loss'][-1]))
                elif self.algorithm == 'anchor':
                    logger.info("Actions statistics: |\t grad norm = %.3f \t value = %.3f \t avg_reward = %.3f \t derivative_loss =  %.3f \t value_loss =  %.3f|" % (bbo_results['grad_norm'][-1], bbo_results['value'][-1], avg_reward, bbo_results['derivative_loss'][-1], bbo_results['value_loss'][-1]))
                logger.info("Best observe      : |\t %f \t \tPi_evaluate: = %f|" % (best_observe, pi_evaluate))

                if (self.algorithm in ['value', 'spline']) and (self.action_space == 1):
                    self.value_vs_f_one_d(n)

            # log to tensorboard
            if args.tensorboard:
                self.writer.add_scalar('evaluation/divergence', divergence, n)
                if self.algorithm in ['value', 'anchor']:
                    self.writer.add_scalars('evaluation/value_reward', {'value': bbo_results['value'][-1], 'pi_evaluate': pi_evaluate, 'best': best_observe}, n)
                    self.writer.add_scalar('evaluation/value_loss', bbo_results['value_loss'][-1], n)
                if self.algorithm in ['first_order', 'second_order', 'anchor']:
                    self.writer.add_scalar('evaluation/grad_norm', bbo_results['grad_norm'][-1], n)
                    self.writer.add_scalar('evaluation/derivative_loss', bbo_results['derivative_loss'][-1], n)
                self.writer.add_scalars('evaluation/pi_evaluate_observe', {'evaluate': pi_evaluate, 'best': best_observe}, n)

                for i in range(len(pi)):
                    self.writer.add_scalars('evaluation/pi_' + str(i), {'pi': pi[i], 'explore': pi_explore[i]}, n)

                if hasattr(self.agent, "pi_net"):
                    self.writer.add_histogram("evaluation/pi_net", self.agent.pi_net.pi.clone().cpu().data.numpy(), n, 'fd')
                if hasattr(self.agent, "value_net"):
                    for name, param in self.agent.value_net.named_parameters():
                        self.writer.add_histogram("evaluation/value_net/%s" % name, param.clone().cpu().data.numpy(), n, 'fd')

        print("End BBO evaluation")
        self.compare_beta_evaluate()

        if self.action_space == 2:
            self.plot_2D_contour()
        return divergence

    def value_vs_f_one_d(self, n):
        path_res = os.path.join(consts.baseline_dir, '1D', '1D_index_{}.pkl'.format(self.iter_index))
        with open(path_res, 'rb') as handle:
            res = pickle.load(handle)
            max_f = res['f'].max()
            min_f = res['f'].min()
            value, pi, pi_value, grad = self.agent.get_evaluation_function(res['norm_policy'][:, 0])

            plt.subplot(111)
            plt.plot(res['policy'][:, 0], (res['f'] - min_f)/(max_f - min_f), color='g', markersize=1, label='f')
            plt.plot(res['policy'][:, 0], (value - min_f)/(max_f - min_f), '-o', color='b', markersize=1, label='value')
            plt.plot(5*pi, (pi_value - min_f)/(max_f - min_f), 'X', color='r', markersize=4, label='pi')
            plt.plot(5 * np.tanh(pi - grad), (pi_value - min_f)/(max_f - min_f), 'v', color='c', markersize=4, label='gard')

            plt.title('1D_index_{} - iteration {}'.format(self.iter_index, n))
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()

            path_dir_fig = os.path.join(self.results_dir, '1D_figures', str(self.iter_index))
            if not os.path.exists(path_dir_fig):
                os.makedirs(path_dir_fig)

            path_fig = os.path.join(path_dir_fig, 'iter {}.pdf'.format(n))
            plt.savefig(path_fig)
            plt.close()

    def plot_2D_contour(self):

        path = os.path.join(self.dirs_locks.analysis_dir, str(self.iter_index))
        path_dir = os.path.join(consts.baseline_dir, '2D_Contour')
        path_res = os.path.join(path_dir, '2D_index_{}.npy'.format(self.iter_index))
        res = np.load(path_res).item()

        x = 5*np.load(os.path.join(path, 'policies.npy'))
        x_exp = 5*np.load(os.path.join(path, 'explore_policies.npy'))

        fig, ax = plt.subplots()
        cs = ax.contour(res['x0'], res['x1'], res['z'], 100)
        plt.plot(x_exp[:, 0], x_exp[:, 1], '.', color='r', markersize=1)
        plt.plot(x[:, 0], x[:, 1], '-o', color='b', markersize=1)
        plt.title(path.split('/')[-1])
        fig.colorbar(cs)

        path_dir_fig = os.path.join(self.results_dir, str(self.iter_index))
        if not os.path.exists(path_dir_fig):
            os.makedirs(path_dir_fig)

        path_fig = os.path.join(path_dir_fig, '2D_index_{}.pdf'.format(self.iter_index))
        plt.savefig(path_fig)

        plt.close()


    def compare_beta_evaluate(self):
        min_val, f0 = self.get_min_f0_val()
        path = os.path.join(self.dirs_locks.analysis_dir, str(self.iter_index))
        pi_eval = np.load(os.path.join(path, 'pi_evaluate.npy'))
        pi_best = np.load(os.path.join(path, 'best_observed.npy'))

        plt.subplot(111)

        plt.loglog(np.arange(len(pi_eval)), (pi_eval - min_val)/(f0 - min_val), color='b', label='pi_evaluate')
        plt.loglog(np.arange(len(pi_best)), (pi_best - min_val) / (f0 - min_val), color='r', label='best_observed')

        plt.legend()
        plt.title('dim = {} index = {} ----- best vs eval'.format(self.action_space, self.iter_index))
        plt.grid(True, which='both')

        path_dir_fig = os.path.join(self.results_dir, str(self.iter_index))
        if not os.path.exists(path_dir_fig):
            os.makedirs(path_dir_fig)

        path_fig = os.path.join(path_dir_fig, 'BestVsEval: dim = {} index = {}.pdf'.format(self.action_space, self.iter_index))
        plt.savefig(path_fig)

        plt.close()

    def get_min_f0_val(self):
        if self.action_space == 1:
            min_val = 0
            f0 = 1
        else:
            min_df = pd.read_csv(os.path.join(consts.baseline_dir, 'min_val.csv'))
            tmp_df = min_df[(min_df.dim == self.action_space) & (min_df.iter_index == self.iter_index)]

            min_val = float(tmp_df.min_val)
            f0 = float(tmp_df.f0)
        return min_val, f0
