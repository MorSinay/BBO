import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

try: import cocoex
except: pass

try: import cma
except: pass

import scipy.optimize
import pandas as pd
import numpy as np
import pwd
import os
from tqdm import tqdm
from collections import defaultdict
from vae import VaeProblem, VAE
from environment import EnvCoco, EnvOneD, EnvVae
from environment import one_d_change_dim
import pickle
username = pwd.getpwuid(os.geteuid()).pw_name
from config import Consts

# if username == 'morsi':
#     base_dir = os.path.join('/Users', username, 'Desktop')
# else:
#     from vae import VaeProblem, VAE
#     base_dir = os.path.join('/data/', username, 'gan_rl')

epsilon = 1

filter_mod = 1

def compare_problem_baseline(dim, index, budget=1000, sub_budget=100):

    if dim == 784:
        problem = VaeProblem(index)
    else:
        suite_name = "bbob"
        suite_filter_options = ("dimensions: "+str(max(dim, 2)))
        suite = cocoex.Suite(suite_name, "", suite_filter_options)

    optimization_function = [scipy.optimize.fmin_slsqp, scipy.optimize.fmin, scipy.optimize.fmin_cobyla, scipy.optimize.fmin_powell,
                             scipy.optimize.fmin_cg, scipy.optimize.fmin_bfgs, cma.fmin2]

    data = defaultdict(list)
    for fmin in optimization_function:
        if dim == 784:
            problem.reset(index)
            env = EnvVae(problem, index)
        elif dim == 1:
            suite.reset()
            problem = suite.get_problem(index)
            env = EnvOneD(problem,index,  False)
        else:
            suite.reset()
            problem = suite.get_problem(index)
            env = EnvCoco(problem, index, False)

        func = env.f

        x = env.initial_solution
        f = []
        eval = []
        best_f = []
        try:
            run_problem(fmin, func, x, budget)
            best_f, f, x = env.get_observed_and_pi_list()

        except:
            x = env.initial_solution
            f.append(func(x))
            best_f.append(f[0])

        data['fmin'].append(fmin.__name__)
        data['index'].append(problem.index)
        data['hit'].append(problem.final_target_hit)
        data['x'].append(x)
        data['id'].append(env.get_problem_id())
        assert min(best_f) == problem.best_observed_fvalue1, "best_observed_error"
        data['best_observed'].append(problem.best_observed_fvalue1)
        data['number_of_evaluations'].append(problem.evaluations)
        data['eval'].append(problem.evaluations)
        data['f'].append(f)
        data['best_list'].append(best_f)
        data['f0'].append(func(env.initial_solution))

        # print(data['fmin'][-1] + ":\tevaluations: " + str(data['number_of_evaluations'][-1]) + "\tf(x): "
        #                                                   + str(func(x)) + "\thit: " + str(data['hit'][-1]))

    data['min_opt'] = (len(data['best_observed'])*[min(data['best_observed'])])
    df = pd.DataFrame(data)
    title = 'dim_{} index_{}'.format(dim, index)
    save_dir = os.path.join(Consts.baseline_dir, 'compare', 'D_{}'.format(dim))
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass
    fmin_file = os.path.join(save_dir, title + '.pkl')
    with open(fmin_file, 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

def run_problem(fmin,  problem, x0, budget):

        fmin_name = fmin.__name__

        if fmin_name is 'fmin_slsqp':
            x, best_val, _, _, _ = fmin(problem, x0, iter=budget, full_output=True, iprint=-1)

        elif fmin_name is 'fmin':
            x, best_val, _, eval_num, _ = fmin(problem, x0, maxfun=budget, disp=False, full_output=True)

        elif fmin_name is 'fmin2':
            x, _ = fmin(problem, x0, 2, {'maxfevals': budget, 'verbose':-9}, restarts=0)

        elif fmin_name is 'fmin_cobyla':
            x = fmin(problem, x0, cons=lambda x: None, maxfun=budget, disp=0, rhoend=1e-9)

        elif fmin_name is 'fmin_powell':
            x, best_val, _, _, eval_num, _ = fmin(problem, x0, maxiter=budget, full_output=1)

        elif fmin_name is 'fmin_cg':
            x, best_val, eval_num, _, _ = fmin(problem, x0, maxiter=budget, full_output=1)

        elif fmin_name is 'fmin_bfgs':
            x, best_val, _, _, eval_num, _, _ = fmin(problem, x0, maxiter=budget, full_output=1)

        else:
            raise NotImplementedError

        return x


def create_copy_file(prefix, dim, index):
    if username == 'morsi':
        assert False, "create_copy_file"

    local_path = os.path.join('/Users', 'morsi', 'Desktop', 'baseline', 'analysis')
    res_dir = Consts.outdir
    dirs = os.listdir(res_dir)

    create_dirs = []
    copy_dirs = []

    dim_len = int(np.log10(dim)) + 2

    for dir in dirs:
        if dir.startswith(prefix) and dir.endswith(str(dim)):
            dir_name = dir[len(prefix)+1:-dim_len]
            create_dirs.append('mkdir {}/{}/{}/{}'.format(local_path, prefix, dim, dir_name))
            copy_dirs.append('scp -r yoda:{}/{}/analysis/{} {}/{}/{}/{}/'.format(res_dir, dir, index, local_path, prefix, dim, dir_name))
        else:
            continue

    with open('copy_file.txt', 'w') as f:
        f.write('rm -r {}/{}/{}\n'.format(local_path, prefix, dim))
        f.write('mkdir {}/{}\n'.format(local_path, prefix))
        f.write('mkdir {}/{}/{}\n'.format(local_path, prefix, dim))
        for line in create_dirs:
            f.write(line)
            f.write('\n')
        for line in copy_dirs:
            f.write(line)
            f.write('\n')


def treeD_plot(problem_index):
    suite = cocoex.Suite("bbob", "", ("dimensions: 2"))
    problem = suite.get_problem(problem_index)
    upper_bound = problem.upper_bounds
    lower_bound = problem.lower_bounds
    interval = 0.1
    res_list = []
    for x0 in np.arange(lower_bound[0], upper_bound[0]+interval, interval):
        for x1 in np.arange(lower_bound[1], upper_bound[1]+interval, interval):
            x = np.array([x0, x1])
            fx = problem(x)
            res_list.append(np.concatenate([x, fx], axis=None))

    res = np.stack(res_list)
    res_dir = os.path.join(Consts.baseline_dir, 'f_eval', '2D')
    if not os.path.exists(res_dir):
        try:
            os.makedirs(res_dir)
        except:
            pass
    path_res = os.path.join(res_dir, '2D_index_{}.npy'.format(problem_index))

    np.save(path_res, res)

def calc_f0():
    save_dir = Consts.baseline_dir
    data_dict = defaultdict(list)
    for dim in [2,3,5,10,20,40]:
        suite = cocoex.Suite("bbob", "", ("dimensions: {}".format(dim)))

        for problem_index in range (360):
            problem = suite.get_problem(problem_index)
            data_dict['id'].append(problem.id)
            data_dict['f0'].append(problem(problem.initial_solution))

    df = pd.DataFrame(data_dict)
    file = os.path.join(save_dir, 'f0.csv')
    df.to_csv(file)


def treeD_plot_contour(problem_index):
    suite = cocoex.Suite("bbob", "", ("dimensions: 2"))
    problem = suite.get_problem(problem_index)
    upper_bound = problem.upper_bounds
    lower_bound = problem.lower_bounds
    interval = 0.1

    x0 = np.arange(lower_bound[0], upper_bound[0] + interval, interval)
    x1 = np.arange(lower_bound[1], upper_bound[1] + interval, interval)
    x0, x1 = np.meshgrid(x0, x1)
    z = np.zeros(x0.shape)

    for i in range(x0.shape[0]):
        for j in range(x1.shape[1]):
            x = np.array([x0[i,j], x1[i,j]])
            z[i,j] = problem(x)

    res_dir = os.path.join(Consts.baseline_dir, 'f_eval', '2D_Contour')
    if not os.path.exists(res_dir):
        try:
            os.makedirs(res_dir)
        except:
            pass


    path_res = os.path.join(res_dir, '2D_index_{}.npy'.format(problem_index))
    np.save(path_res, {'x0':x0, 'x1':x1, 'z':z})


def D1_plot(problem_index):
    suite = cocoex.Suite("bbob", "", ("dimensions: 2"))
    problem = suite.get_problem(problem_index)

    upper_bound = problem.upper_bounds
    lower_bound = problem.lower_bounds
    interval = 0.001

    x0 = np.arange(-1, 1 + interval, interval)
    norm_policy = np.clip(one_d_change_dim(x0), -1, 1)
    f = np.zeros(x0.shape)
    policy = 0.5 * (norm_policy + 1) * (upper_bound - lower_bound) + lower_bound

    for i in range(x0.shape[0]):
        f[i] = problem(policy[i])

    res_dir = os.path.join(Consts.baseline_dir, 'f_eval', '1D')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    path_res = os.path.join(res_dir, '1D_index_{}.pkl'.format(problem_index))
    with open(path_res, 'wb') as handle:
        pickle.dump({'norm_policy':norm_policy, 'policy':policy, 'f':f}, handle, protocol=pickle.HIGHEST_PROTOCOL)

def nD_plot(dim, problem_index):
    suite = cocoex.Suite("bbob", "", ("dimensions: {}".format(dim)))
    problem = suite.get_problem(problem_index)

    upper_bound = problem.upper_bounds
    lower_bound = problem.lower_bounds
    interval = 0.001

    norm_policy = np.arange(-1, 1 + interval, interval).reshape(-1,1)
    norm_policy = np.repeat(norm_policy, dim, axis=1)
    f = np.zeros(norm_policy.shape[0])
    policy = 0.5 * (norm_policy + 1) * (upper_bound - lower_bound) + lower_bound

    for i in range(policy.shape[0]):
        f[i] = problem(policy[i])

    res_dir = os.path.join(Consts.baseline_dir, 'f_eval', '{}D'.format(dim))
    if not os.path.exists(res_dir):
        try:
            os.makedirs(res_dir)
        except:
            pass

    path_res = os.path.join(res_dir, '{}D_index_{}.pkl'.format(dim, problem_index))

    with open(path_res, 'wb') as handle:
        pickle.dump({'norm_policy':norm_policy, 'policy':policy, 'f':f}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def visualization(problem_index):
    suite = cocoex.Suite("bbob", "", ("dimensions: 2"))
    problem = suite.get_problem(problem_index)
    upper_bound = problem.upper_bounds
    lower_bound = problem.lower_bounds
    interval = 0.1

    #fig, (ax1, ax2) = plt.subplots(1, 2)
    fig = plt.figure()
    fig.suptitle('index_{} -- {}'.format(problem_index, problem.id))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, projection='3d')
    ax3 = plt.subplot(313)

    res_dir = os.path.join(Consts.baseline_dir, 'visualization')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    x0 = np.arange(lower_bound[0], upper_bound[0] + interval, interval)
    x1 = np.arange(lower_bound[1], upper_bound[1] + interval, interval)
    x0, x1 = np.meshgrid(x0, x1)
    z = np.zeros(x0.shape)

    for i in range(x0.shape[0]):
        for j in range(x1.shape[1]):
            x = np.array([x0[i, j], x1[i, j]])
            z[i, j] = problem(x)

    ax1.contour(x0, x1, z, 100)

    f_list = []
    x_list = []
    for x0 in np.arange(lower_bound[0], upper_bound[0] + interval, interval):
        for x1 in np.arange(lower_bound[1], upper_bound[1] + interval, interval):
            x = np.array([x0, x1])
            fx = problem(x)
            f_list.append(fx)
            x_list.append(x)

    f_list = np.array(f_list)
    x_list = np.array(x_list)
    #ax2 = fig.gca(projection='3d')
    ax2.plot_trisurf(x_list[:, 0], x_list[:, 1], f_list, cmap='winter')

    interval = 0.0001
    x0 = np.arange(-1, 1 + interval, interval)
    norm_policy = np.clip(one_d_change_dim(x0), -1, 1)
    f = np.zeros(x0.shape)
    policy = 0.5 * (norm_policy + 1) * (upper_bound - lower_bound) + lower_bound

    for i in range(x0.shape[0]):
        f[i] = problem(policy[i])

    ax3.plot(policy, f, color='g', markersize=1, label='f')
    ax3.set_xlabel('x')
    ax3.set_ylabel('f(x)')
    ax3.grid(True, which='both')

    path_fig = os.path.join(res_dir, 'index_{:03d}.pdf'.format(problem_index))
    plt.savefig(path_fig)
    plt.close()


def vaeD_plot(problem_index):
    problem = VaeProblem(problem_index)

    upper_bound = problem.upper_bounds
    lower_bound = problem.lower_bounds
    interval = 0.0001

    norm_policy = np.arange(-1, 1 + interval, interval).reshape(-1,1)
    norm_policy = np.repeat(norm_policy, problem.dimension, axis=1)
    f = np.zeros(norm_policy.shape[0])
    policy = 0.5 * (norm_policy + 1) * (upper_bound - lower_bound) + lower_bound

    for i in range(policy.shape[0]):
        f[i] = problem.func(policy[i])

    res_dir = os.path.join(Consts.baseline_dir, 'f_eval', '{}D'.format(problem.dimension))
    if not os.path.exists(res_dir):
        try:
            os.makedirs(res_dir)
        except:
            pass

    path_res = os.path.join(res_dir, '{}D_index_{}.pkl'.format(problem.dimension, problem_index))

    with open(path_res, 'wb') as handle:
        pickle.dump({'norm_policy':norm_policy, 'policy':policy, 'f':f}, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_baseline_cmp(dim, index):
    path_res = os.path.join(Consts.baseline_dir, 'compare', 'D_{}'.format(dim), 'dim_{} index_{}'.format(dim, index) + ".pkl")
    with open(path_res, 'rb') as handle:
        res = pickle.load(handle)
        return res

    return None

def merge_baseline_one_line_compare(dims=[1, 2, 3, 5, 10, 20, 40, 784]):

    data = defaultdict(list)
    for dim in dims:
        for index in tqdm(range(0, 360, filter_mod)):
            optimizer_res = get_baseline_cmp(dim, index)

            data['dim'].append(dim)
            data['iter_index'].append(index)
            data['f0'].append(optimizer_res['f0'][0])
            data['id'].append(optimizer_res['id'][0])
            data['min_opt'].append(optimizer_res['min_opt'][0])
            for i, op in enumerate(optimizer_res['fmin']):
                res = optimizer_res[optimizer_res['fmin'] == op]
                data[op + '_best_observed'].append(float(res['best_observed']))
                data[op + '_budget'].append(float(res['number_of_evaluations']))

    df = pd.DataFrame(data)
    file = os.path.join(Consts.baseline_dir, 'compare.csv')
    df.to_csv(file)

def run_baseline(dims=[1, 2, 3, 5, 10, 20, 40]):

    for dim in dims:
        for i in tqdm(range(0, 360, filter_mod)):
            compare_problem_baseline(dim, i, budget=150000)

    merge_baseline_one_line_compare(dims)

def avg_dim_best_observed(dim, save_file, prefix):

    max_len = -1
    res_dir = Consts.outdir
    dirs = os.listdir(res_dir)

    compare_dirs = defaultdict()

    dim_len = int(np.log10(dim)) + 2
    indexes = set([str(x) for x in range(360)])
    for dir in dirs:
        if dir.startswith(prefix) and dir.endswith(str(dim)):
            alg = dir[len(prefix) + 1: -dim_len]
            compare_dirs[alg] = os.path.join(res_dir, dir, 'analysis')
            dir_index = os.listdir(compare_dirs[alg])
            indexes = indexes.intersection(dir_index)
        else:
            continue

    data = defaultdict(list)
    for index in indexes:
        optimizer_res = get_baseline_cmp(dim, index)
        min_val = optimizer_res['min_opt'][0] - 0.0001
        f0 = optimizer_res['f0'][0]


        for i, op in enumerate(optimizer_res['fmin']):
            res = optimizer_res[optimizer_res['fmin'] == op]
            arr = np.array(res['best_list'][i])
            arr = (arr - min_val) / f0
            data[op].append(arr)
            max_len = max(max_len, len(data[op][-1]))

        for key, path in compare_dirs.items():
            try:
                pi_best = np.load(os.path.join(path, index, 'best_list_with_explore.npy'))
                pi_best = (pi_best - min_val) / f0
            except:
                pi_best = np.ones(1)

            data[key].append(pi_best)
            max_len = max(max_len, len(pi_best))

    plt.subplot(111)
    for j, (key, l) in enumerate(data.items()):
        i = 0
        list_sum = np.zeros(max_len)
        for x in l:
            i+=1
            lastval = x[-1]
            list_sum += np.concatenate([x, lastval * np.ones(max_len-len(x))])
        list_sum /= i
        plt.loglog(np.arange(max_len), list_sum, color=Consts.color[j], label=key)

    plt.legend(loc='upper right')
    plt.grid(True, which="both")
    plt.title("AVG BEST OBSERVED COMPARE - DIM {}".format(dim))
    path_fig = os.path.join(Consts.baseline_dir, save_file)
    plt.savefig(path_fig)
    plt.close()

def avg_dim_dist_from_best_x(dim, save_file, prefix):

    max_len = -1
    res_dir = Consts.outdir
    dirs = os.listdir(res_dir)

    compare_dirs = defaultdict()

    dim_len = int(np.log10(dim)) + 2
    indexes = set([str(x) for x in range(360)])
    for dir in dirs:
        if dir.startswith(prefix) and dir.endswith(str(dim)):
            alg = dir[len(prefix) + 1: -dim_len]
            compare_dirs[alg] = os.path.join(res_dir, dir, 'analysis')
            dir_index = os.listdir(compare_dirs[alg])
            indexes = indexes.intersection(dir_index)
        else:
            continue

    data = defaultdict(list)
    for index in indexes:
        for key, path in compare_dirs.items():
            try:
                dist_x = np.load(os.path.join(path, index, 'dist_x.npy'))
            except:
                continue

            data[key].append(dist_x)
            max_len = max(max_len, len(dist_x))

    if not len(data):
        return

    plt.subplot(111)
    for j, (key, l) in enumerate(data.items()):
        i = 0
        list_sum = np.zeros(max_len)
        for x in l:
            i+=1
            lastval = x[-1]
            list_sum += np.concatenate([x, lastval * np.ones(max_len-len(x))])
        list_sum /= i
        plt.loglog(np.arange(max_len), list_sum, color=Consts.color[j], label=key)

    plt.legend(loc='upper right')
    plt.grid(True, which="both")
    plt.title("AVG DIST FROM BEST X COMPARE - DIM {}".format(dim))
    path_fig = os.path.join(Consts.baseline_dir, save_file)
    plt.savefig(path_fig)
    plt.close()

def merge_bbo(optimizers=[], dimension=[1, 2, 3, 5, 10, 20, 40, 784], save_file='baseline_cmp.pdf', plot_sum=False):
    compare_file = os.path.join(Consts.baseline_dir, 'compare.csv')
    assert os.path.exists(compare_file), 'no compare file'

    baseline_df = pd.read_csv(os.path.join(compare_file))
    res_dir = os.path.join(Consts.baseline_dir, 'results')

    for op in optimizers:
        op_df = pd.read_csv(os.path.join(res_dir, op, op+'_{}'.format(dimension[0]) + ".csv"))

        for dim in dimension[1:]:
            df_d = pd.read_csv(os.path.join(res_dir, op, op+'_{}'.format(dim) + ".csv"))
            op_df = op_df.append(df_d, ignore_index=True)

        op_df = op_df[['id', 'best_observed', 'number_of_evaluations']]
        op_df = op_df.rename(columns={"best_observed": op+"_best_observed",
                                "number_of_evaluations": op+"_number_of_evaluations"})

        baseline_df = baseline_df.merge(op_df, on='id')

    columns = baseline_df.columns
    best_observed = []
    for item in columns:
        if item.endswith('_best_observed'):
            best_observed.append(item)

    baseline_df['min_val'] = baseline_df[best_observed].min(axis=1)

    file = os.path.join(Consts.baseline_dir, 'tmp_compare.csv')
    baseline_df.to_csv(file)

    X = [20*i for i in range(len(dimension))]
    ax = plt.subplot(111)
    w = 1

    for j, best_observed_op in enumerate(best_observed):
        res = []
        x = [X[i] + j*w for i in range(len(dimension))]
        for dim in dimension:
            temp_df = baseline_df[baseline_df.dim == dim]
            if plot_sum:
                compare_method = (temp_df[best_observed_op].values - temp_df['min_val'].values) / np.abs(temp_df['f0'].values)
            else:
                compare_method = np.abs(temp_df[best_observed_op].values - temp_df['min_val'].values) < epsilon
            count = len(compare_method)
            res.append(compare_method.sum()/count)
        optim = best_observed_op[:-len('_best_observed')]
        ax.bar(x, res, width=w, color=Consts.color[j], align='center', label=optim)

    ax.set_xticks([i + len(dimension)//2 for i in X])
    ax.set_xticklabels(dimension)
   # ax.autoscale(tight=True)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
    ax.legend()
    #plt.title("BASELINE COMPARE - BUDGET {}".format(max_budget))
    path_fig = os.path.join(res_dir, save_file)
    plt.savefig(path_fig)
    plt.close()


def bbo_evaluate_compare(dim, index, prefix='CMP'):

    optimizer_res = get_baseline_cmp(dim, index)
    min_val = optimizer_res['min_opt'][0] - 0.0001
    f0 = optimizer_res['f0'][0]

    res_dir = Consts.outdir
    dirs = os.listdir(res_dir)

    compare_dirs = defaultdict()

    dim_len = int(np.log10(dim)) + 2

    for dir in dirs:
        if dir.startswith(prefix) and dir.endswith(str(dim)):
            alg = dir[len(prefix)+1 : -dim_len]
            compare_dirs[alg] = os.path.join(res_dir, dir, 'analysis', str(index))
        else:
            continue

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('compare dim = {} index = {}'.format(dim, index))

    colors = Consts.color

    x_max = -1
    for i, key in enumerate(compare_dirs.keys()):
        path = compare_dirs[key]
        try:
            pi_eval = np.load(os.path.join(path, 'pi_evaluate.npy'))
            pi_best = np.load(os.path.join(path, 'best_observed.npy'))

            ax1.loglog(np.arange(len(pi_eval)), (pi_eval - min_val)/(f0 - min_val), color=colors[i], label=key)
            ax2.loglog(np.arange(len(pi_best)), (pi_best - min_val) / (f0 - min_val), color=colors[i])

            x_max = max(x_max, len(pi_eval), len(pi_best))
        except:
            pass

    fig.legend(loc='lower left', prop={'size': 6}, ncol=3)
    ax1.grid(True, which='both')
    ax2.grid(True, which='both')
    ax1.set_title('reward_pi_evaluate')
    ax2.set_title('best_observed')
    ax1.set_xlim([10, x_max])
    ax2.set_xlim([10, x_max])

    path_fig = os.path.join(Consts.baseline_dir, 'Compare bbo - dim = {} index = {}.pdf'.format(dim, index))
    plt.savefig(path_fig)

    plt.close()

def get_best_solution(dim, index):
    optimizer_res = get_baseline_cmp(dim, index)
    best_idx = optimizer_res['best_observed'].values.argmin()
    best_array = np.array(optimizer_res['best_list'][best_idx])
    best_x_idx = best_array.argmin()
    x = optimizer_res['x'][best_idx][best_x_idx]
    best_val = best_array[best_x_idx]

    assert (min(x) < -5 or max(x) > 5), "out of range - get_best_solution dim {} problem {}".format(dim, index)
    return x, best_val

if __name__ == '__main__':
    #merge_baseline_one_line_compare(dims=[1, 2, 3, 5, 10, 20, 40])

    optimizers = ['first_order_unconstrained', 'first_order_cone_beu']
    dims = [1, 2, 3, 5, 10, 20, 40]
    merge_bbo(optimizers=optimizers, dimension=dims, save_file='baseline_cmp_success.pdf', plot_sum=False)
    merge_bbo(optimizers=optimizers, dimension=dims, save_file='baseline_cmp_avg_sum.pdf', plot_sum=True)

    # # # bbo_evaluate_compare(dim=40, index=15, prefix='CMP')
    # #

    dims = [40]

    for dim in dims:
        prefix = 'RUN'
        fig_name = '{} dim {} best observed avg.pdf'.format(prefix,dim)
        avg_dim_best_observed(dim=dim, save_file=fig_name, prefix=prefix)
        fig_name = '{} dim {} dist avg.pdf'.format(prefix, dim)
        avg_dim_dist_from_best_x(dim=dim, save_file=fig_name, prefix=prefix)

    # # for i in tqdm(range(0, 360, 1)):
    # #     visualization(i)
    #
    # #treeD_plot_contour(0)
    # #calc_f0()
    # #twoD_plot_contour(index)
    #
    # # dim = 40
    # # for i in tqdm(range(0, 360, filter_mod)):
    # #     compare_problem_baseline(dim, i, budget=150000)
    #
