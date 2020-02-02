import argparse
import time
import socket
import os
import pwd
import fcntl

parser = argparse.ArgumentParser(description='gan_rl')
username = pwd.getpwuid(os.geteuid()).pw_name
server = socket.gethostname()

if "gpu" in server:
    base_dir = os.path.join('/home/mlspeech/', username, 'data/gan_rl', server)
elif "root" == username:
    base_dir = r'/workspace/data/gan_rl/'
else:
    base_dir = os.path.join('/data/', username, 'gan_rl')

def boolean_feature(feature, default, help):

    global parser
    featurename = feature.replace("-", "_")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--%s' % feature, dest=featurename, action='store_true', help=help)
    feature_parser.add_argument('--no-%s' % feature, dest=featurename, action='store_false', help=help)
    parser.set_defaults(**{featurename: default})


# General Arguments
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--batch', type=int, default=1024, help='Mini-Batch Size')
parser.add_argument('--n-explore', type=int, default=64, help='exploration')

# strings
parser.add_argument('--game', type=str, default='bbo', help='bbo | net')
parser.add_argument('--identifier', type=str, default='debug', help='The name of the model to use')
parser.add_argument('--algorithm', type=str, default='EGL', help='[EGL | value | second_order]')

boolean_feature('debug', False, 'debug flag')
boolean_feature('spline', False, 'spline net')
boolean_feature('hassian', False, 'Use hassian in gradient net')
boolean_feature('regularization', False, 'regularization')
boolean_feature('trust-region', True, 'use trust region')

#boolean_feature('vae', False, 'run vae problem')
# VAE parameters
parser.add_argument('--latent', type=int, default=10, help='Size of latent space')

# # booleans
boolean_feature("load-last-model", False, 'Load the last saved model')
boolean_feature("tensorboard", False, "Log results to tensorboard")
boolean_feature('importance-sampling', False, "Derivative eval")
parser.add_argument('--grad-clip', type=float, default=0, help='grad clipping')
parser.add_argument('--vae', type=str, default='gaussian', help='gaussian | uniform')
parser.add_argument('--budget', type=int, default=150000, help='Number of steps')
# parameters
parser.add_argument('--resume', type=int, default=-1, help='Resume experiment number, set -1 for last experiment')

# #exploration parameters
parser.add_argument('--epsilon', type=float, default=0.1, help='exploration parameter before behavioral period')
parser.add_argument('--cone-angle', type=float, default=2, help='cone angle - default pi/3')
parser.add_argument('--norm', type=str, default='robust_scaler', help='normalization option - min_max | mean | mean_std')
parser.add_argument('--explore', type=str, default='ball', help='exploration option - ball | cone | rand')
boolean_feature("best-explore-update", True, 'move to the best value of exploration')
parser.add_argument('--trust-region-con', type=int, default=10, help='Trust Region Condition')
parser.add_argument('--min-iter', type=int, default=40, help='Minimum iteration')
parser.add_argument('--agent', type=str, default='trust', help='Agent type - trust|robust|single')

#
# #dataloader
parser.add_argument('--cpu-workers', type=int, default=24, help='How many CPUs will be used for the data loading')
parser.add_argument('--cuda-default', type=int, default=0, help='Default GPU')
#
# #train parameters
parser.add_argument('--printing-interval', type=int, default=50, help='Number of exploration steps between printing results')
parser.add_argument('--replay-updates-interval', type=int, default=50, help='Number of training iterations between q-target updates')
parser.add_argument('--replay-memory-factor', type=int, default=32, help='Replay factor')
parser.add_argument('--warmup-minibatch', type=int, default=5, help='Warm up batches')
parser.add_argument('--trust-factor', type=float, default=0.9, help='Warm up factor')
parser.add_argument('--r-norm-alg', type=str, default='log', help='log |relu | tanh | none')
parser.add_argument('--epsilon-factor', type=float, default=0.97, help='Epsilon factor')
parser.add_argument('--learn-iteration', type=int, default=60, help='Learning iteration')
parser.add_argument('--alpha', type=float, default=0.5, help='moving avg factor')
parser.add_argument('--loss', type=str, default='huber', help='derivative loss huber|mse')
parser.add_argument('--start', type=int, default=0, help='')
parser.add_argument('--filter', type=int, default=15, help='')


#
# #actors parameters
parser.add_argument('--problem-index', type=int, default=-1, help='Problem Index or -1 for all')
parser.add_argument('--robust-scaler-lr', type=float, default=1e-1, help='robust learning rate')
parser.add_argument('--pi-lr', type=float, default=1e-2, help='pi learning rate')
parser.add_argument('--value-lr', type=float, default=1e-3, help='value learning rate')
parser.add_argument('--action-space', type=int, default=10, help='Problem dimension')
parser.add_argument('--layer', type=int, default=256, help='Value hidden layer size')
parser.add_argument('--seed', type=int, default=0, help='Set seed')
parser.add_argument('--exploration', type=str, default='GRAD', help='GRAD|UNIFORM')

# distributional learner

args = parser.parse_args()


# consts
class Consts(object):

    server = socket.gethostname()
    start_time = time.time()
    exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    nop = 0

    mem_threshold = int(2e9)

    outdir = os.path.join(base_dir, 'results')
    baseline_dir = os.path.join(base_dir, 'baseline')
    logdir = os.path.join(base_dir, 'logs')
    vaedir = os.path.join(base_dir, 'vae_bbo')

    if not os.path.exists(logdir):
        try:
            os.makedirs(logdir)
        except:
            pass
    if not os.path.exists(outdir):
        try:
            os.makedirs(outdir)
        except:
            pass

    # color = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'lime', 'gold', 'slategray', 'indigo', 'maroon', 'plum', 'pink', 'tan', 'khaki', 'silver',
    #          'navy', 'skyblue', 'teal', 'darkkhaki', 'indianred', 'orchid', 'lightgrey', 'dimgrey']

    color = ['r', 'dodgerblue', 'green', 'darkorange', 'mediumpurple', 'peru', 'pink', 'orchid', 'lightslategrey', 'gold', 'turquoise', 'lime', 'slategray', 'indigo', 'maroon', 'plum', 'khaki', 'silver',
             'navy', 'skyblue', 'teal', 'darkkhaki', 'indianred', 'lightgrey', 'dimgrey']

consts = Consts()



class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DirsAndLocksSingleton(metaclass=Singleton):
    def __init__(self, exp_name):

        self.outdir = consts.outdir
        self.exp_name = exp_name
        self.root = os.path.join(self.outdir, self.exp_name)

        self.tensorboard_dir = os.path.join(self.root, 'tensorboard')
        self.checkpoints_dir = os.path.join(self.root, 'checkpoints')
        self.results_dir = os.path.join(self.root, 'results')
        self.code_dir = os.path.join(self.root, 'code')
        self.analysis_dir = os.path.join(self.root, 'analysis')
        self.checkpoint = os.path.join(self.checkpoints_dir, 'checkpoint')

        try:
            if not os.path.exists(self.tensorboard_dir):
                os.makedirs(self.tensorboard_dir)
            if not os.path.exists(self.checkpoints_dir):
                os.makedirs(self.checkpoints_dir)
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            if not os.path.exists(self.code_dir):
                os.makedirs(self.code_dir)
            if not os.path.exists(self.analysis_dir):
                os.makedirs(self.analysis_dir)
        except:
            pass
