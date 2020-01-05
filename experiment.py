import time
import os
import sys
import numpy as np
import torch
from tensorboardX import SummaryWriter
from single_agent import BBOAgent
from robust_agent import RobustAgent
from trust_region_agent import TrustRegionAgent
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
from visualize_2d import get_baseline_cmp

import scipy.optimize  # to define the solver to be benchmarked

class Experiment(object):

    def __init__(self, logger_file, env):

        # parameters
        self.action_space = args.action_space
        dirs = os.listdir(consts.outdir)
        self.n_explore = args.n_explore
        self.load_model = args.load_last_model
        self.load_last = args.load_last_model
        self.resume = args.resume
        self.env = env
        self.problem_id = self.env.get_problem_id()
        self.algorithm = args.algorithm
        self.iter_index = env.problem_iter
        self.printing_interval = args.printing_interval

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
        self.tensorboard_dir = os.path.join(self.dirs_locks.tensorboard_dir, str(self.iter_index))
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

    def select_agent(self):
        agent_type = args.agent
        if agent_type == 'trust':
            return TrustRegionAgent
        elif agent_type == 'robust':
            return RobustAgent
        elif agent_type == 'single':
            return BBOAgent
        else:
            raise NotImplementedError

    def bbo(self):
        self.agent = self.select_agent()(self.exp_name, self.env, checkpoint=self.checkpoint)

        player = self.agent.minimize()
        divergence = 0

        for n, bbo_results in (enumerate(player)):
            pi = bbo_results['policies'][-1].cpu().numpy()
            pi_explore = torch.mean(bbo_results['explore_policies'][-1], dim=0).cpu().numpy()
            pi_evaluate = bbo_results['reward_pi_evaluate'][-1]

            avg_reward = np.average(bbo_results['rewards'][-1])
            best_observe = bbo_results['best_observed'][-1]
            divergence = bbo_results['divergence'][-1]

            if not n % self.printing_interval:
                logger.info("---------------- iteration: {} - Problem ID :{} ---------------".format(n, self.problem_id))
                logger.info("Problem iter index     :{}\t\tDim: {}\tDivergence: {}".format(self.iter_index, self.action_space, divergence))
                if self.algorithm in ['first_order', 'second_order']:
                    logger.info("Actions statistics: |\t grad norm = %.3f \t avg_reward = %.3f| \t derivative_loss =  %.3f" % (bbo_results['grad_norm'][-1], avg_reward, bbo_results['derivative_loss'][-1]))
                elif self.algorithm == ['value']:
                    logger.info("Actions statistics: |\t value = %.3f \t avg_reward = %.3f \t value_loss =  %.3f|" % (bbo_results['value'][-1], avg_reward, bbo_results['value_loss'][-1]))
                elif self.algorithm == 'anchor':
                    logger.info("Actions statistics: |\t grad norm = %.3f \t value = %.3f \t avg_reward = %.3f \t derivative_loss =  %.3f \t value_loss =  %.3f|" % (bbo_results['grad_norm'][-1], bbo_results['value'][-1], avg_reward, bbo_results['derivative_loss'][-1], bbo_results['value_loss'][-1]))
                logger.info("Best observe      : |\t %f \t Pi_evaluate: = %f| \tBest_pi_evaluate: = %f" % (best_observe, pi_evaluate, bbo_results['best_pi_evaluate'][-1]))
                logger.info("dist_x            : |\t %f \t dist_f: = %f|" % (bbo_results['dist_x'][-1], bbo_results['dist_f'][-1]))

                if args.debug and self.algorithm in ['value', 'anchor']:
                    self.value_vs_f_eval(n)

                if args.debug and self.algorithm in ['first_order', 'second_order', 'anchor']:
                    self.grad_norm_on_f_eval(n)

            # log to tensorboard
            if args.tensorboard:
                self.writer.add_scalar('evaluation/divergence', divergence, n)
                if self.algorithm in ['value', 'anchor']:
                    self.writer.add_scalars('evaluation/value_reward', {'value': bbo_results['value'][-1], 'reward_pi_evaluate': pi_evaluate, 'best': best_observe}, n)
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
        self.compare_pi_evaluate()

        if self.action_space == 2:
            self.plot_2D_contour()
        return divergence

    def value_vs_f_eval(self, n):
        path_res = os.path.join(consts.baseline_dir, 'f_eval', '{}D'.format(self.action_space), '{}D_index_{}.pkl'.format(self.action_space, self.iter_index))
        with open(path_res, 'rb') as handle:
            res = pickle.load(handle)
            norm_policy = res['norm_policy']
            policy = res['norm_policy']
            f = res['f']

            if self.action_space == 1:
                norm_policy = norm_policy[:, 0, np.newaxis]

            value, pi, pi_value, pi_with_grad, policy_grads, norm_f = self.agent.get_evaluation_function(norm_policy, f)

            pi = pi.reshape(-1, 1)

            plt.subplot(111)
            plt.plot(policy[:, 0], norm_f, color='g', markersize=1, label='f')
            plt.plot(policy[:, 0], value, '-o', color='b', markersize=1, label='value')
            plt.plot(policy[:, 0], policy_grads, 'H', color='m', markersize=1, label='norm_grad')
            plt.plot(5*pi[0], pi_value, 'X', color='r', markersize=4, label='pi')
            plt.plot(5*pi_with_grad[0], pi_value, 'v', color='c', markersize=4, label='gard')

            plt.title('value net for alg {} - {}D_index_{} - iteration {}'.format(self.algorithm, self.action_space, self.iter_index, n))
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()

            path_dir_fig = os.path.join(self.results_dir, str(self.iter_index))
            if not os.path.exists(path_dir_fig):
                os.makedirs(path_dir_fig)

            path_fig = os.path.join(path_dir_fig, 'value iter {}.pdf'.format(n))
            plt.savefig(path_fig)
            plt.close()

    def grad_norm_on_f_eval(self, n):
        path_res = os.path.join(consts.baseline_dir, 'f_eval', '{}D'.format(self.action_space), '{}D_index_{}.pkl'.format(self.action_space, self.iter_index))
        with open(path_res, 'rb') as handle:
            res = pickle.load(handle)
            norm_policy = res['norm_policy']
            policy = res['norm_policy']
            f = res['f']

            if self.action_space == 1:
                norm_policy = norm_policy[:, 0, np.newaxis]

            grad_direct, pi, pi_grad_norm, pi_with_grad, norm_f = self.agent.get_grad_norm_evaluation_function(norm_policy, f)

            num_grad = (norm_f[1:] - norm_f[:-1]) / (np.linalg.norm(norm_policy[1:] - norm_policy[:-1], axis=1) + 1e-5)

            pi = pi.reshape(-1, 1)
            pi_with_grad = pi_with_grad.reshape(-1, 1)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(policy[:-1, 0], norm_f[:-1], color='g', markersize=1, label='f')
            ax2.plot(policy[:-1, 0], num_grad, '-o', color='r', markersize=1, label='grad_numerical')
            ax2.plot(policy[:-1, 0], grad_direct, '-o', color='b', markersize=1, label='grad_norm')

            ax2.plot(5*pi[0], pi_grad_norm, 'X', color='r', markersize=4, label='pi')
            ax2.plot(5*pi_with_grad[0], pi_grad_norm, 'v', color='c', markersize=4, label='pi_with_grad')

            fig.suptitle('derivative net for alg {} - {}D_index_{} - iteration {}'.format(self.algorithm, self.action_space, self.iter_index, n))
            fig.legend()

            path_dir_fig = os.path.join(self.results_dir, str(self.iter_index))
            if not os.path.exists(path_dir_fig):
                os.makedirs(path_dir_fig)

            path_fig = os.path.join(path_dir_fig, 'derivative iter {}.pdf'.format(n))
            fig.savefig(path_fig)
            plt.close()

    def plot_2D_contour(self):

        path = os.path.join(self.dirs_locks.analysis_dir, str(self.iter_index))
        path_dir = os.path.join(consts.baseline_dir, 'f_eval', '2D_Contour')
        path_res = os.path.join(path_dir, '2D_index_{}.npy'.format(self.iter_index))
        res = np.load(path_res).item()

        x = 5*np.load(os.path.join(path, 'policies.npy'))
        x_exp = 5*np.load(os.path.join(path, 'explore_policies.npy'))

        fig, ax = plt.subplots()
        cs = ax.contour(res['x0'], res['x1'], res['z'], 100)
        plt.plot(x_exp[:, 0], x_exp[:, 1], '.', color='r', markersize=1)
        plt.plot(x[:, 0], x[:, 1], '-o', color='b', markersize=1)
        plt.title('alg {} - 2D_Contour index {}'.format(self.algorithm, self.iter_index))
        fig.colorbar(cs)

        path_dir_fig = os.path.join(self.results_dir, str(self.iter_index))
        if not os.path.exists(path_dir_fig):
            os.makedirs(path_dir_fig)

        path_fig = os.path.join(path_dir_fig, '2D_index_{}.pdf'.format(self.iter_index))
        plt.savefig(path_fig)

        plt.close()

    def compare_pi_evaluate(self):
        optimizer_res = get_baseline_cmp(self.action_space, self.iter_index)
        min_val = optimizer_res['min_opt'][0] - 0.0001
        f0 = optimizer_res['f0'][0]

        path = os.path.join(self.dirs_locks.analysis_dir, str(self.iter_index))
        pi_eval = np.load(os.path.join(path, 'observed_list_with_explore.npy'))
        pi_best = np.load(os.path.join(path, 'best_list_with_explore.npy'))

        plt.subplot(111)

        colors = consts.color
        #plt.loglog(np.arange(len(rewards)), (rewards - min_val) / (f0 - min_val), linestyle='None', markersize=1, marker='o', color=colors[2], label='explore')
        plt.loglog(np.arange(len(pi_eval)), (pi_eval - min_val)/(f0 - min_val), color=colors[0], label='reward_pi_evaluate')
        plt.loglog(np.arange(len(pi_best)), (pi_best - min_val) / (f0 - min_val), color=colors[1], label='best_observed')

        for i, op in enumerate(optimizer_res['fmin']):
            res = optimizer_res[optimizer_res['fmin'] == op]
            op_eval = np.array(res['f'].values[0])
            op_eval = np.clip(op_eval, a_max=f0, a_min=-np.inf)
            plt.loglog(np.arange(len(op_eval)), 1 + (op_eval - min_val) / (f0 - min_val), color=colors[3+i], label=op)

        plt.legend()
        plt.title('alg {} - dim = {} index = {} ----- best vs eval'.format(self.algorithm, self.action_space, self.iter_index))
        plt.grid(True, which='both')

        path_dir_fig = os.path.join(self.results_dir, str(self.iter_index))
        if not os.path.exists(path_dir_fig):
            os.makedirs(path_dir_fig)

        path_fig = os.path.join(path_dir_fig, 'BestVsEval - dim = {} index = {}.pdf'.format(self.action_space, self.iter_index))
        plt.savefig(path_fig)

        plt.close()

    # def get_min_f0_val(self):
    #     if self.action_space == 1:
    #         min_val = 0
    #         f0 = 1
    #     else:
    #         min_df = pd.read_csv(os.path.join(consts.baseline_dir, 'min_val.csv'))
    #         tmp_df = min_df[(min_df.dim == self.action_space) & (min_df.iter_index == self.iter_index)]
    #
    #         min_val = float(tmp_df.min_val)
    #         f0 = float(tmp_df.f0)
    #     return min_val, f0
