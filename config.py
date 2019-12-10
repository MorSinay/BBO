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

parser.add_argument('--batch', type=int, default=30, help='Mini-Batch Size')

# strings
parser.add_argument('--game', type=str, default='bbo', help='bbo | net')
parser.add_argument('--identifier', type=str, default='debug', help='The name of the model to use')
parser.add_argument('--algorithm', type=str, default='first_order', help='[first_order | value | second_order]')

boolean_feature('debug', False, 'debug flag')
#boolean_feature('vae', False, 'run vae problem')

# # booleans
boolean_feature("load-last-model", False, 'Load the last saved model')
boolean_feature("tensorboard", False, "Log results to tensorboard")
boolean_feature('importance-sampling', False, "Derivative eval")
boolean_feature('bandage', False, "Bandage")
boolean_feature('grad-clip', False, "Clip gradients")
parser.add_argument('--vae', type=str, default='gaussian', help='gaussian | uniform')
parser.add_argument('--budget', type=int, default=10000, help='Number of steps')
# parameters
parser.add_argument('--resume', type=int, default=-1, help='Resume experiment number, set -1 for last experiment')

# #exploration parameters
parser.add_argument('--epsilon', type=float, default=0.1, metavar='ε', help='exploration parameter before behavioral period')
parser.add_argument('--explore', type=str, default='grad_direct', metavar='explore', help='exploration option - grad_rand | grad_direct | rand')
parser.add_argument('--update-step', type=str, default='n_step', metavar='update', help='beta update step - n_step | best_step | first_vs_last | no_update')
boolean_feature("best-explore-update", False, 'move to the best value of exploration')
parser.add_argument('--grad-steps', type=int, default=10, metavar='grad', help='Gradient step')
parser.add_argument('--stop-con', type=int, default=200, metavar='stop', help='Stopping Condition')
parser.add_argument('--clip', type=float, default=1., metavar='clip', help='Gradient Clipping')

#
# #dataloader
parser.add_argument('--cpu-workers', type=int, default=24, help='How many CPUs will be used for the data loading')
parser.add_argument('--cuda-default', type=int, default=0, help='Default GPU')
#
# #train parameters
parser.add_argument('--checkpoint-interval', type=int, default=1000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--replay-updates-interval', type=int, default=50, metavar='STEPS', help='Number of training iterations between q-target updates')
parser.add_argument('--replay-memory-factor', type=int, default=10, help='Replay factor')
parser.add_argument('--delta', type=float, default=0.1, metavar='delta', help='Total variation constraint')
parser.add_argument('--drop', type=float, default=0, metavar='drop out', help='Drop out')
#
# #actors parameters
parser.add_argument('--problem-index', type=int, default=-1, help='Problem Index or -1 for all')
parser.add_argument('--beta-lr', type=float, default=1e-2, metavar='LR', help='beta learning rate')
parser.add_argument('--value-lr', type=float, default=1e-3, metavar='LR', help='value learning rate')
parser.add_argument('--action-space', type=int, default=10, metavar='dimension', help='Problem dimension')
parser.add_argument('--layer', type=int, default=128, metavar='layer', help='Value hidden layer size')
parser.add_argument('--seed', type=int, default=150, metavar='seed', help='Set seed')
parser.add_argument('--exploration', type=str, default='GRAD', metavar='N', help='GRAD|UNIFORM')

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
    logdir = os.path.join(base_dir, 'logs')

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
