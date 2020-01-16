import argparse
import time
import numpy as np
import socket
import os
import pwd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter
from distutils.dir_util import copy_tree
import sys
import torch
import pandas as pd
import shutil
from loguru import logger
import random
from collections import defaultdict

project_name = 'landmarks'
username = pwd.getpwuid(os.geteuid()).pw_name

if "gpu" in socket.gethostname():
    home_dir = os.path.join('/home/dsi/', username)
else:
    home_dir = os.path.join('/home/mlspeech/', username)

predictor_path = os.path.join(home_dir,
                              'projects/landmarks/facial_landmarks_recognition/shape_predictor_68_face_landmarks.dat')


def boolean_feature(feature, default, help):

    global parser
    featurename = feature.replace("-", "_")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--%s' % feature, dest=featurename, action='store_true', help=help)
    feature_parser.add_argument('--no-%s' % feature, dest=featurename, action='store_false', help=help)
    parser.set_defaults(**{featurename: default})

parser = argparse.ArgumentParser(description=project_name)
# Arguments

# global parameters
parser.add_argument('--dataset-dir', type=str, default='/localdata/elads/celeba/', help='Directory of the CelebA Dataset')

parser.add_argument('--generator-dir', type=str,
                    default='/home/dsi/elads/data/bbo/results/pagan_store_80000_iter_d_2_celeba_exp_0000_20191208_1404301',
                    help='The generator/discriminator models')
parser.add_argument('--classifier-dir', type=str,
                    default='/home/dsi/elads/data/bbo/results/attribute_debug_att_head_balanced_celeba_exp_0000_20191209_215136',
                    help='The classifier model')

parser.add_argument('--predictor-path', type=str, default=predictor_path, help='Path for the landmark predictor')
parser.add_argument('--algorithm', type=str, default='egl', help='[egl|igl]')

parser.add_argument('--num', type=int, default=-1, help='Resume experiment number, set -1 for new experiment')
parser.add_argument('--cpu-workers', type=int, default=48, help='How many CPUs will be used for the data loading')
parser.add_argument('--cuda', type=int, default=0, help='GPU Number')
parser.add_argument('--identifier', type=str, default='debug', help='The name of the model to use')

# booleans

boolean_feature("optimize", False, 'Optimization routine')
boolean_feature("multi-gpu", False, 'Split batch over all GPUs')
boolean_feature("tensorboard", True, "Log results to tensorboard")
boolean_feature("reload", False, "Load saved model")
boolean_feature("half", True, 'Use half precision calculation')
boolean_feature("reshape", False, 'Reshape image size')
boolean_feature("resnet", True, 'Resnet Architecture')
boolean_feature("lognet", False, 'Log networks data to tensorboard')

# experiment parameters

parser.add_argument('--dataset', type=str, default='celeba', help='Dataset name [full|mini]')
parser.add_argument('--seed', type=int, default=0, help='Seed for reproducability')
parser.add_argument('--image-size', type=int, default=64, help='Size of image after reshape')
parser.add_argument('--height', type=int, default=218, help='Image Height')
parser.add_argument('--width', type=int, default=178, help='Image width')
# parser.add_argument('--height', type=int, default=512, help='Image Height')
# parser.add_argument('--width', type=int, default=512, help='Image width')


parser.add_argument('--penalty', type=float, default=10., help='Penalty for no face detection')

parser.add_argument('--weight-decay', type=float, default=0., help='L2 regularization coefficient')
parser.add_argument('--clip', type=float, default=1., help='Clip Gradient L2 norm')

parser.add_argument('--epochs', type=int, default=100, metavar='STEPS', help='Total number of backward steps')
parser.add_argument('--train-epoch', type=int, default=50, metavar='BATCHES', help='Length of each epoch (in batches)')
parser.add_argument('--test-epoch', type=int, default=10, metavar='BATCHES', help='Length of test epoch (in batches)')
parser.add_argument('--batch', type=int, default=32, help='Batch Size')

# # booleans
boolean_feature("load-last-model", False, 'Load the last saved model')
boolean_feature("grad", False, 'Use grad net')
boolean_feature('importance-sampling', False, "Derivative eval")
boolean_feature("best-explore-update", True, 'move to the best value of exploration')

# #exploration parameters
parser.add_argument('--epsilon', type=float, default=0.2, metavar='ε', help='exploration parameter before behavioral period')
parser.add_argument('--explore', type=str, default='rand', metavar='explore', help='exploration option - grad_rand | grad_guided | rand')
parser.add_argument('--update-step', type=str, default='n_step', metavar='update', help='pi update step - n_step | best_step | first_vs_last | no_update')
parser.add_argument('--grad-steps', type=int, default=8, metavar='grad', help='Gradient step')

# #train parameters
parser.add_argument('--replay-memory-factor', type=int, default=10, help='Replay factor')
parser.add_argument('--delta', type=float, default=0.1, metavar='delta', help='Total variation constraint')
parser.add_argument('--dropout', type=float, default=0., help='Dropout regularization coefficient')
parser.add_argument('--channel', type=int, default=32, help='Channel multiplier')

#
# #actors parameters
parser.add_argument('--problem-index', type=int, default=-1, help='Problem Index or -1 for random problem')
parser.add_argument('--pi-lr', type=float, default=1e-2, metavar='LR', help='pi learning rate')
parser.add_argument('--value-lr', type=float, default=1e-3, metavar='LR', help='value learning rate')
parser.add_argument('--action-space', type=int, default=512, metavar='dimension', help='Problem dimension')
parser.add_argument('--layer', type=int, default=256, help='Channel multiplier')
parser.add_argument('--exploration', type=str, default='GRAD', metavar='N', help='GRAD|UNIFORM')

parser.add_argument('--stop-con', type=int, default=40, help='Stopping Condition')
parser.add_argument('--n-explore', type=int, default=32, help='exploration')
parser.add_argument('--epsilon-factor', type=float, default=0.9, help='Epsilon factor')
parser.add_argument('--warmup-minibatch', type=int, default=2, help='Warm up batches')
parser.add_argument('--warmup-factor', type=int, default=1, help='Warm up factor')
parser.add_argument('--cone-angle', type=float, default=3, help='cone angle - default pi/3')
parser.add_argument('--grad-clip', type=float, default=0, help='grad clipping')
parser.add_argument('--learn-iteration', type=int, default=60, help='Learning iteration')
parser.add_argument('--loss', type=str, default='huber', help='derivative loss huber|mse')
parser.add_argument('--trust-factor', type=float, default=0.9, help='Warm up factor')


args = parser.parse_args()
seed = args.seed


def set_seed(seed=seed):

    if 'cnt' not in set_seed.__dict__:
        set_seed.cnt = 0
    set_seed.cnt += 1

    if seed is None:
        seed = args.seed * set_seed.cnt

    if seed > 0:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Experiment(object):

    def __init__(self):

        set_seed()

        torch.set_num_threads(100)
        logger.info("Welcome to: Fake Voice Generator")
        logger.info(' ' * 26 + 'Simulation Hyperparameters')
        for k, v in vars(args).items():
            logger.info(' ' * 26 + k + ': ' + str(v))

        # consts

        self.uncertainty_samples = 1
        # parameters

        self.start_time = time.time()
        self.exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.device = torch.device("cuda:%d" % args.cuda)
        self.opt_level = "O1"  if args.half else "O0"

        if "gpu" in socket.gethostname():
            self.root_dir = os.path.join('/home/dsi/', username, 'data', project_name)
        elif "root" == username:
            self.root_dir = os.path.join('/workspace/data', project_name)
        else:
            self.root_dir = os.path.join('/data/', username, project_name)

        self.base_dir = os.path.join(self.root_dir, 'results')
        self.data_dir = os.path.join(self.root_dir, 'data', args.dataset)

        for folder in [self.base_dir, self.root_dir, self.data_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        dirs = os.listdir(self.base_dir)

        self.resume = args.num
        temp_name = "%s_%s_%s_exp" % (args.algorithm, args.identifier, args.dataset)
        self.exp_name = ""
        self.load_model = True
        if self.resume >= 0:
            for d in dirs:
                if "%s_%04d_" % (temp_name, self.resume) in d:
                    self.exp_name = d
                    self.exp_num = self.resume
                    break

        if not self.exp_name:
            # count similar experiments
            n = max([-1] + [int(d.split("_")[-3]) for d in dirs if temp_name in d]) + 1
            self.exp_name = "%s_%04d_%s" % (temp_name, n, self.exptime)
            self.exp_num = n
            self.load_model = False

        logger.info(f"Experience name: {self.exp_name}")

        # init experiment parameters
        self.root = os.path.join(self.base_dir, self.exp_name)

        # set dirs
        self.tensorboard_dir = os.path.join(self.root, 'tensorboard')
        self.checkpoints_dir = os.path.join(self.root, 'checkpoints')
        self.results_dir = os.path.join(self.root, 'results')
        self.code_dir = os.path.join(self.root, 'code')
        self.checkpoint = os.path.join(self.checkpoints_dir, 'checkpoint')

        if self.load_model and args.reload:
            logger.info("Resuming existing experiment")

        else:

            if not self.load_model:
                logger.info("Creating new experiment")

            else:
                logger.info("Deleting old experiment")
                shutil.rmtree(self.root)

            os.makedirs(self.root)
            os.makedirs(self.tensorboard_dir)
            os.makedirs(self.checkpoints_dir)
            os.makedirs(self.results_dir)
            os.makedirs(self.code_dir)

            # make log dirs
            os.makedirs(os.path.join(self.results_dir, 'train'))
            os.makedirs(os.path.join(self.results_dir, 'eval'))

            # copy code to dir
            copy_tree(os.path.dirname(os.path.realpath(__file__)), self.code_dir)

            # write args to file
            filename = os.path.join(self.root, "args.txt")
            with open(filename, 'w') as fp:
                fp.write('\n'.join(sys.argv[1:]))

            pd.to_pickle(vars(args), os.path.join(self.root, "args.pkl"))

        self.detector = os.path.join(self.data_dir, '..', "shape_predictor_68_face_landmarks.dat")

        if args.dataset == 'celeba':
            self.dataset_dir = os.path.join(args.dataset_dir, 'cropped')
            self.attributes_file = os.path.join(args.dataset_dir, 'list_attr_celeba.csv')
        else:
            raise NotImplementedError

        # initialize tensorboard writer
        if args.tensorboard:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir, comment=args.identifier)

    def log_data(self, train_results, test_results=None, n=0, alg=None):

        defaults_argv = defaultdict(dict)

        for param, val in train_results['scalar'].items():
            if type(val) is dict:
                for p, v in val.items():
                    val[p] = np.mean(v)
            else:
                train_results['scalar'][param] = np.mean(val)

        if test_results is not None:
            for param, val in test_results['scalar'].items():
                if type(val) is dict:
                    for p, v in val.items():
                        val[p] = np.mean(v)
                else:
                    test_results['scalar'][param] = np.mean(val)

        if args.tensorboard:

            if alg is not None:
                networks = alg.get_networks()
                for net in networks:
                    for name, param in networks[net]():
                        try:
                            self.writer.add_histogram("weight_%s/%s" % (net, name), param.data.cpu().numpy(), n,
                                                      bins='tensorflow')
                            self.writer.add_histogram("grad_%s/%s" % (net, name), param.grad.cpu().numpy(), n,
                                                      bins='tensorflow')
                            if hasattr(param, 'intermediate'):
                                self.writer.add_histogram("iterm_%s/%s" % (net, name), param.intermediate.cpu().numpy(),
                                                          n,
                                                          bins='tensorflow')
                        except:
                            pass

            for log_type in train_results:
                log_func = getattr(self.writer, f"add_{log_type}")
                for param in train_results[log_type]:

                    if type(train_results[log_type][param]) is dict:
                        for p, v in train_results[log_type][param].items():
                            log_func(f"train_{param}/{p}", v, n, **defaults_argv[log_type])
                    else:
                        log_func(f"train/{param}", train_results[log_type][param], n, **defaults_argv[log_type])

            if test_results is not None:
                for log_type in test_results:
                    log_func = getattr(self.writer, f"add_{log_type}")
                    for param in test_results[log_type]:

                        if type(test_results[log_type][param]) is dict:
                            for p, v in test_results[log_type][param].items():
                                log_func(f"eval_{param}/{p}", v, n, **defaults_argv[log_type])
                        else:
                            log_func(f"eval/{param}", test_results[log_type][param], n, **defaults_argv[log_type])

        stat_line = 'Train: '
        for param in train_results['scalar']:
            if type(train_results['scalar'][param]) is not dict:
                stat_line += '  %s %g \t|' % (param, train_results['scalar'][param])
        logger.info(stat_line)

        if test_results is not None:
            stat_line = 'Eval: '
            for param in test_results['scalar']:
                if type(test_results['scalar'][param]) is not dict:
                    stat_line += '  %s %g \t|' % (param, test_results['scalar'][param])
            logger.info(stat_line)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if args.tensorboard:
            self.writer.export_scalars_to_json(os.path.join(self.tensorboard_dir, "all_scalars.json"))
            self.writer.close()


exp = Experiment()
