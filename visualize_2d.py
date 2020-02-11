from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
import matplotlib.ticker as mtick

import torch
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

epsilon = 0.1
filter_mod = 15
optimization_function = {'trust-ncg': scipy.optimize.minimize,
                         'trust-constr': scipy.optimize.minimize,
                         'trust-exact': scipy.optimize.minimize,
                         'trust-krylov': scipy.optimize.minimize,
                         'slsqp': scipy.optimize.fmin_slsqp,
                         'fmin': scipy.optimize.fmin,
                         'cobyla': scipy.optimize.fmin_cobyla,
                         'powell': scipy.optimize.fmin_powell,
                         'cg': scipy.optimize.fmin_cg,
                         'bfgs': scipy.optimize.fmin_bfgs,
                         'cma': cma.fmin2
}
optimization_colors = { 'fmin': 'green',
                        'slsqp': 'dodgerblue',
                        'powell': 'mediumpurple',
                        'cg': 'peru',
                        'cobyla': 'darkorange',
                        'bfgs': 'pink',
                        'cma': 'orchid',
                        'IGL': 'gold',
                        'EGL': 'turquoise'
}
problem_name_set = ['Sphere', 'Ellipsoid separable', 'Rastrigin separable', 'Skew Rastrigin-Bueche separ',
               'Linear slope', 'Attractive sector', 'Step-ellipsoid', 'Rosenbrock original',
               'Rosenbrock rotated', 'Ellipsoid', 'Discus', 'Bent cigar', 'Sharp ridge', 'Sum of different powers',
               'Rastrigin', 'Weierstrass', 'Schaffer F7, condition 10', 'Schaffer F7, condition 1000', 'Griewank-Rosenbrock F8F2',
               'Schwefel x*sin(x)', 'Gallagher 101 peaks', 'Gallagher 21 peaks', 'ats ras', 'Lunacek bi-Rastrigin']


def compare_problem_baseline(dim, index, budget=1000, sub_budget=100):

    if dim == 784:
        problem = VaeProblem(index)
    else:
        suite_name = "bbob"
        suite_filter_options = ("dimensions: "+str(max(dim, 2)))
        suite = cocoex.Suite(suite_name, "", suite_filter_options)

    data = defaultdict(list)
    for alg, fmin in optimization_function.items():
        if dim == 784:
            problem.reset(index)
            env = EnvVae(problem, index, to_numpy=False)
        elif dim == 1:
            suite.reset()
            problem = suite.get_problem(index)
            env = EnvOneD(problem, index, need_norm=False, to_numpy=False)
        else:
            suite.reset()
            problem = suite.get_problem(index)
            env = EnvCoco(problem, index, need_norm=False, to_numpy=False)

        func = env.f

        x = env.initial_solution
        f = []
        eval = []
        best_f = []
        try:
            if dim==784 and alg in ['cg','cma']:
                run_problem(alg, fmin, func, x, budget)
                best_f, f, x = env.get_observed_and_pi_list()
            else:
                raise NotImplementedError
        except:
            env.samples = 0
            x = env.initial_solution
            _ = func(x)
            best_f, f, x = env.get_observed_and_pi_list()

        data['fmin'].append(alg)
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

def run_problem(alg, fmin,  problem, x0, budget):

    if alg == 'slsqp':
        x, best_val, _, _, _ = fmin(problem, x0, iter=budget, full_output=True, iprint=-1)

    elif alg == 'fmin':
        x, best_val, _, eval_num, _ = fmin(problem, x0, maxfun=budget, disp=False, full_output=True)

    elif alg == 'cma':
        x, _ = fmin(problem, x0, 2, {'maxfevals': budget, 'verbose':-9}, restarts=0)

    elif alg == 'cobyla':
        x = fmin(problem, x0, cons=lambda x: None, maxfun=budget, disp=0, rhoend=1e-9)

    elif alg == 'powell':
        x, best_val, _, _, eval_num, _ = fmin(problem, x0, maxiter=budget, full_output=1)

    elif alg == 'cg':
        x, best_val, eval_num, _, _ = fmin(problem, x0, maxiter=budget, full_output=1)

    elif alg == 'bfgs':
        x, best_val, _, _, eval_num, _, _ = fmin(problem, x0, maxiter=budget, full_output=1)

    elif alg == 'trust-ncg':
        _ = fmin(problem, x0, args=(), method='trust-ncg', jac='cs', hess='cs', options={'maxiter': budget, 'disp': False})

    elif alg == 'trust-constr':
        _ = fmin(problem, x0, args=(), method='trust-constr', jac='2-point', hess='2-point', options={'maxiter': budget, 'disp': False})

    elif alg == 'trust-exact':
        _ = fmin(problem, x0, args=(), method='trust-exact', jac='2-point', hess='2-point', options={'maxiter': budget, 'disp': False})

    elif alg == 'trust-krylov':
        _ = fmin(problem, x0, args=(), method='trust-krylov', jac='2-point', hess='2-point', options={'maxiter': budget, 'disp': False})

    else:
        raise NotImplementedError


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
    interval = 0.001
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
    df.to_csv(file, index=False)


def treeD_plot_contour(problem_index):
    suite = cocoex.Suite("bbob", "", ("dimensions: 2"))
    problem = suite.get_problem(problem_index)
    upper_bound = problem.upper_bounds
    lower_bound = problem.lower_bounds
    interval = 0.001

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
    interval = 0.01

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

    ax3.plot(policy, f, color='g', markersize=2, linewidth=4, label='f')
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

    for dim in dims:
        data = defaultdict(list)
        for index in tqdm(range(0, 360, 1)):
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
    df.to_csv(file, index=False)


def run_baseline(dims=[1, 2, 3, 5, 10, 20, 40]):

    for dim in dims:
        for i in tqdm(range(0, 360, filter_mod)):
            compare_problem_baseline(dim, i, budget=150000)

    merge_baseline_one_line_compare(dims)

def avg_dim_best_observed(dim, save_file, alg_name_list, prefix_list, with_op=False, axs=None, y_label=False):

    max_len = 150000
    res_dir = Consts.outdir
    dirs = os.listdir(res_dir)

    compare_dirs = defaultdict()

    indexes = set([str(x) for x in range(360)])
    for dir in dirs:
        for i, prefix in enumerate(prefix_list):
            if dir.startswith(prefix) and dir.endswith(str(dim)):
                alg = alg_name_list[i]
                compare_dirs[alg] = os.path.join(res_dir, dir, 'analysis')
                dir_index = os.listdir(compare_dirs[alg])
                tmp_id = []
                for id in dir_index:
                    try:
                        _ = np.load(os.path.join(compare_dirs[alg], id, 'best_list_with_explore.npy'), allow_pickle=True)
                    except:
                        continue
                    tmp_id.append(id)
                indexes = indexes.intersection(set(tmp_id))
            else:
                continue

    data = defaultdict(list)
    for index in tqdm(indexes):
        optimizer_res = get_baseline_cmp(dim, index)
        min_val = optimizer_res['min_opt'][0]
        f0 = optimizer_res['f0'][0]

        for key, path in compare_dirs.items():
            try:
                pi_best = np.load(os.path.join(path, index, 'best_list_with_explore.npy'),  allow_pickle=True)
                min_val = min(min_val, pi_best.min())
            except:
                pass

        min_val -= 1e-5
        if with_op:
            for op in optimization_function.keys():
                if op.startswith('trust'):
                    continue
                res = optimizer_res[optimizer_res['fmin'] == op]
                i = (optimizer_res['fmin'] == op).idxmax()
                arr = np.array(res['best_list'][i])
                arr = np.clip(arr, a_max=f0, a_min=min_val)
                arr = arr[:max_len]
                arr = np.concatenate([arr, arr[-1] * np.ones(max_len - len(arr))])
                arr = (arr - min_val) / (f0 - min_val + 1e-5)
                data[op].append(arr)

        for key in alg_name_list:
            try:
                pi_best = np.load(os.path.join(compare_dirs[key], index, 'best_list_with_explore.npy'),  allow_pickle=True)
                pi_best = np.clip(pi_best, a_max=f0, a_min=min_val)
                pi_best = pi_best[:max_len]
                pi_best = np.concatenate([pi_best, pi_best[-1] * np.ones(max_len - len(pi_best))])
                pi_best = (pi_best - min_val) / (f0 - min_val + 1e-5)
            except:
                continue

            data[key].append(pi_best)

    if axs is None:
        axs = plt.subplot(111)
        axs.set_title("AVG BEST OBSERVED COMPARE - DIM {}".format(dim))
        path_fig = os.path.join(Consts.baseline_dir, save_file)
    else:
        path_fig = None

    if y_label:
        for j, key in enumerate(optimization_colors.keys()):
            l = data[key]
            i = 0
            list_sum = np.zeros(max_len)
            for x in l:
                i+=1
                list_sum += x
            list_sum /= i

            if dim == 784 and key in ['fmin', 'slsqp', 'cobyla', 'powell', 'bfgs']:
                continue


            if key == 'fmin':
                axs.loglog(np.arange(max_len)+1, list_sum, color=optimization_colors[key], linewidth=4, label='Nelder Mead', basex=2, basey=2)
            elif key == 'cma':
                axs.loglog(np.arange(max_len) + 1, list_sum, color=optimization_colors[key], linewidth=4, label='CMA-ES', basex=2, basey=2 )
            elif key == 'Perturb':
                axs.loglog(np.arange(max_len) + 1, list_sum, color=Consts.color[0], linestyle='dashed', linewidth=4, label=key.upper(), basex=2, basey=2)
            elif optimization_colors.get(key, 0):
                axs.loglog(np.arange(max_len) + 1, list_sum, color=optimization_colors[key], linewidth=4, label=key.upper(), basex=2, basey=2)
            else:
                axs.loglog(np.arange(max_len) + 1, list_sum, color=Consts.color[j+1], linewidth=4, label=key.upper(), basex=2, basey=2)
    else:
        for j, (key, l) in enumerate(data.items()):
            i = 0
            list_sum = np.zeros(max_len)
            for x in l:
                i+=1
                list_sum += x
            list_sum /= i

            if dim == 784 and key in ['fmin', 'slsqp', 'cobyla', 'powell', 'bfgs']:
                continue


            if key == 'fmin':
                axs.loglog(np.arange(max_len)+1, list_sum, color=optimization_colors[key], linewidth=4, label='Nelder Mead', basex=2, basey=2)
            elif key == 'cma':
                axs.loglog(np.arange(max_len) + 1, list_sum, color=optimization_colors[key], linewidth=4, label='CMA-ES', basex=2, basey=2 )
            elif key == 'Perturb':
                axs.loglog(np.arange(max_len) + 1, list_sum, color=Consts.color[0], linestyle='dashed', linewidth=4, label=key.upper(), basex=2, basey=2)
            elif optimization_colors.get(key, 0):
                axs.loglog(np.arange(max_len) + 1, list_sum, color=optimization_colors[key], linewidth=4, label=key.upper(), basex=2, basey=2)
            else:
                axs.loglog(np.arange(max_len) + 1, list_sum, color=Consts.color[j+1], linewidth=4, label=key.upper(), basex=2, basey=2)

    axs.grid(True, which="both")
    axs.axis('tight')
    axs.set_xlabel('t\na', fontsize=24)
    axs.tick_params(labelsize=22)

    if y_label:
        axs.set_ylabel(r'$\overline{\Delta y}_{best}^t$', fontsize=24)
    else:
        axs.set_yticklabels([])
        #pass


    #axs.set_ylim(bottom=2**(-7), top=1.5)

    if path_fig is not None:
        axs.legend(loc='lower left')
        axs.set_xlim(left=384)
        plt.savefig(path_fig)
        plt.close()
    else:
        return axs



def avg_perturb_best_observed(dim, save_file, div_name,alg_name_list, prefix_list, axs=None):

    max_len = 150000
    res_dir = Consts.outdir
    dirs = os.listdir(res_dir)

    compare_dirs = defaultdict()
    div_dirs = defaultdict()

    indexes = set([str(x) for x in range(360)])
    #indexes = set([str(x) for x in range(1)])
    for dir in dirs:
        for i, prefix in enumerate(prefix_list):
            if dir.startswith(prefix) and dir.endswith(str(dim)):
                alg = alg_name_list[i]
                compare_dirs[alg] = os.path.join(res_dir, dir, 'analysis')
                dir_index = os.listdir(compare_dirs[alg])
                tmp_id = []
                for id in dir_index:
                    try:
                        _ = np.load(os.path.join(compare_dirs[alg], id, 'best_list_with_explore.npy'), allow_pickle=True)
                    except:
                        continue
                    tmp_id.append(id)
                indexes = indexes.intersection(set(tmp_id))
            else:
                continue

    data = defaultdict(list)
    for index in tqdm(indexes):
        optimizer_res = get_baseline_cmp(dim, index)
        min_val = optimizer_res['min_opt'][0]
        f0 = optimizer_res['f0'][0]

        for key, path in compare_dirs.items():
            try:
                pi_best = np.load(os.path.join(path, index, 'best_list_with_explore.npy'),  allow_pickle=True)
                min_val = min(min_val, pi_best.min())
            except:
                pass
        for key, path in div_dirs.items():
            try:
                pi_best = np.load(os.path.join(path, index, 'best_list_with_explore.npy'),  allow_pickle=True)
                min_val = min(min_val, pi_best.min())
            except:
                pass

        min_val -= 1e-5
        for key in alg_name_list:
            try:
                pi_best = np.load(os.path.join(compare_dirs[key], index, 'best_list_with_explore.npy'),  allow_pickle=True)
                pi_best = np.clip(pi_best, a_max=f0, a_min=min_val)
                pi_best = pi_best[:max_len]
                pi_best = np.concatenate([pi_best, pi_best[-1] * np.ones(max_len - len(pi_best))])
                pi_best = (pi_best - min_val) / (f0 - min_val + 1e-5)
            except:
                continue

            data[key].append(pi_best)

    if axs is None:
        axs = plt.subplot(111)
        axs.set_title("AVG BEST OBSERVED COMPARE - DIM {}".format(dim))
        path_fig = os.path.join(Consts.baseline_dir, save_file)
    else:
        path_fig = None


    i = 0
    div_sum = np.zeros(max_len)
    for x in data[div_name]:
        i += 1
        div_sum += x
    div_sum /= i

    for j, (key, l) in enumerate(data.items()):
        i = 0
        list_sum = np.zeros(max_len)
        for x in l:
            i+=1
            list_sum += x
        list_sum /= i
        list_sum /= div_sum
        list_sum = pd.Series(list_sum)
        list_sum = 100 * (1 - list_sum.rolling(500, win_type='blackmanharris', min_periods=1, center=True).mean().values)

        if dim == 784 and key in ['fmin', 'slsqp', 'cobyla', 'powell', 'bfgs']:
            continue

        if key == 'fmin':
            axs.plot(np.arange(max_len)+1, list_sum, color=optimization_colors[key], linewidth=4, label='Nelder Mead')
        elif key == 'cma':
            axs.plot(np.arange(max_len) + 1, list_sum, color=optimization_colors[key], linewidth=4, label='CMA-ES')
        elif key == 'Perturb':
            axs.plot(np.arange(max_len) + 1, list_sum, color=Consts.color[0], linestyle='dashed', linewidth=4, label=key.upper())
        elif optimization_colors.get(key, 0):
            axs.plot(np.arange(max_len) + 1, list_sum, color=optimization_colors[key], linewidth=4, label=key.upper())
        else:
            axs.plot(np.arange(max_len) + 1, list_sum, color=Consts.color[j+1], linewidth=4, label=key.upper())

    axs.grid(True, which="both")
    axs.axis('tight')
    axs.set_xlabel('t\na', fontsize=24)
    axs.tick_params(labelsize=22)
    axs.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.xticks([0, 50e3, 100e3, 150e3], ["0", "50K", "100K", "150K"])
    axs.set_xlim(left=384)
    #axs.set_ylim(bottom=2**(-7), top=1.5)

    if path_fig is not None:
        axs.legend(loc='lower left')
        plt.savefig(path_fig)
        plt.close()
    else:
        return axs


def dim_plot(kk, alg_name_list, prefix_list, with_op, ax):
    max_len = 150000
    res_dir = Consts.outdir
    dirs = os.listdir(res_dir)

    compare_dirs = defaultdict()

    indexes = set([str(x + 15 * kk) for x in range(15)])
    for dir in dirs:
        for i, prefix in enumerate(prefix_list):
            if dir.startswith(prefix) and dir.endswith(str(dim)):
                alg = alg_name_list[i]
                compare_dirs[alg] = os.path.join(res_dir, dir, 'analysis')
                dir_index = os.listdir(compare_dirs[alg])
                tmp_id = []
                for id in dir_index:
                    try:
                        _ = np.load(os.path.join(compare_dirs[alg], id, 'best_list_with_explore.npy'), allow_pickle=True)
                    except:
                        continue
                    tmp_id.append(id)
                indexes = indexes.intersection(set(tmp_id))
            else:
                continue

    data = defaultdict(list)
    for index in tqdm(indexes):
        optimizer_res = get_baseline_cmp(dim, index)
        min_val = optimizer_res['min_opt'][0]
        f0 = optimizer_res['f0'][0]

        for key, path in compare_dirs.items():
            pi_best = np.load(os.path.join(path, index, 'best_list_with_explore.npy'), allow_pickle=True)
            min_val = min(min_val, pi_best.min())

        min_val -= 1e-5
        if with_op:
            for op in optimization_function.keys():
                if op.startswith('trust'):
                    continue
                res = optimizer_res[optimizer_res['fmin'] == op]
                i = (optimizer_res['fmin'] == op).idxmax()
                arr = np.array(res['best_list'][i])
                arr = np.clip(arr, a_max=f0, a_min=min_val)
                arr = arr[:max_len]
                arr = np.concatenate([arr, arr[-1] * np.ones(max_len - len(arr))])
                arr = (arr - min_val) / (f0 - min_val + 1e-5)
                data[op].append(arr)

        for key in alg_name_list:
            try:
                pi_best = np.load(os.path.join(compare_dirs[key], index, 'best_list_with_explore.npy'), allow_pickle=True)
                pi_best = np.clip(pi_best, a_max=f0, a_min=min_val)
                pi_best = pi_best[:max_len]
                pi_best = np.concatenate([pi_best, pi_best[-1] * np.ones(max_len - len(pi_best))])
                pi_best = (pi_best - min_val) / (f0 - min_val + 1e-5)
            except:
                continue

            data[key].append(pi_best)

    for j, (key, l) in enumerate(data.items()):
        i = 0
        list_sum = np.zeros(max_len)
        for x in l:
            i += 1
            list_sum += x
        list_sum /= i

        if dim == 784 and key in ['fmin', 'slsqp', 'cobyla', 'powell', 'bfgs']:
            continue

        if key == 'fmin':
            ax.loglog(np.arange(max_len) + 1, list_sum, color=optimization_colors[key], linewidth=4, label='Nelder Mead', basex=2, basey=2)
        elif key == 'cma':
            ax.loglog(np.arange(max_len) + 1, list_sum, color=optimization_colors[key], linewidth=4, label='CMA-ES', basex=2, basey=2)
        elif key == 'Perturb':
            ax.loglog(np.arange(max_len) + 1, list_sum, color=Consts.color[0], linestyle='dashed', linewidth=4, label=key.upper(), basex=2, basey=2)
        elif optimization_colors.get(key, 0):
            ax.loglog(np.arange(max_len) + 1, list_sum, color=optimization_colors[key], linewidth=4, label=key.upper(), basex=2, basey=2)
        else:
            ax.loglog(np.arange(max_len) + 1, list_sum, color=Consts.color[j + 1], linewidth=4, label=key.upper(), basex=2, basey=2)

    ax.grid(True, which="both")
    ax.axis('tight')
    ax.title.set_text('{}: {}'.format(kk+1, problem_name_set[kk]))
    #ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.autoscale(tight=True)

def _2d_plot(index, ax):

    suite = cocoex.Suite("bbob", "", ("dimensions: 2"))
    problem = suite.get_problem(15*index)
    upper_bound = problem.upper_bounds
    lower_bound = problem.lower_bounds
    interval = 0.1

    x0 = np.arange(lower_bound[0], upper_bound[0] + interval, interval)
    x1 = np.arange(lower_bound[1], upper_bound[1] + interval, interval)
    x0, x1 = np.meshgrid(x0, x1)
    z = np.zeros(x0.shape)

    for k in range(x0.shape[0]):
        for j in range(x1.shape[1]):
            x = np.array([x0[k, j], x1[k, j]])
            z[k, j] = problem(x)

    min_val = z.min()
    z -= min_val - 1e-3
    max_val = z.max()
    z /= (max_val + 1e-3)

    ax.contour(x0, x1, np.log(z), 100)
    ax.autoscale(tight=True)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    #ax.axis('equal')

def _3d_plot(index, ax):

    suite = cocoex.Suite("bbob", "", ("dimensions: 2"))
    problem = suite.get_problem(15*index)
    upper_bound = problem.upper_bounds
    lower_bound = problem.lower_bounds
    interval = 0.1

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

    ax.plot_trisurf(x_list[:, 0], x_list[:, 1], f_list, cmap='winter')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

def _1d_plot(index, ax):

    suite = cocoex.Suite("bbob", "", ("dimensions: 2"))
    problem = suite.get_problem(15*index)
    upper_bound = problem.upper_bounds
    lower_bound = problem.lower_bounds

    interval = 0.0001
    x0 = np.arange(-1, 1 + interval, interval)
    norm_policy = np.clip(one_d_change_dim(x0), -1, 1)
    f = np.zeros(x0.shape)
    policy = 0.5 * (norm_policy + 1) * (upper_bound - lower_bound) + lower_bound

    for k in range(x0.shape[0]):
        f[k] = problem(policy[k])

    min_val = f.min()
    f -= min_val - 1e-3
    max_val = f.max()
    f /= (max_val + 1e-3)

    ax.semilogy(policy, f, color='g', markersize=2, linewidth=4, label='log view')
    ax.set_ylabel('log view', color='g')  # we already handled the x-label with ax1

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(r'$f_{1D}$', color=color)  # we already handled the x-label with ax1
    ax2.plot(policy, f, color=color, markersize=2, linewidth=4, label=r'$f_{1D}$')
    ax2.tick_params(axis='y', labelcolor=color)

    ax.autoscale(tight=True)
    ax2.autoscale(tight=True)

    ax.grid(True, which='both')

def avg_dim_problem(dim, save_file, alg_name_list, prefix_list, with_op=False):

    fig = plt.figure(figsize=(22, 16))
    for kk, problem_set in tqdm(enumerate(range(0,360,15))):

        ax = plt.subplot(6, 4, kk+ 1)
        dim_plot(kk, alg_name_list, prefix_list, with_op, ax)
        if kk == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(-0.6, 1))

    path_fig = os.path.join(Consts.baseline_dir, save_file + '.pdf')
    #fig.suptitle("AVG BEST OBSERVED COMPARE - DIM {}".format(dim), fontsize=20)
    plt.savefig(path_fig)
    plt.close()

def avg_dim_problem_2d(dim, save_file, alg_name_list, prefix_list, with_op=False):

    # fig, axs = plt.subplots(3, 4, figsize=(16, 8))
    for kk, problem_set in tqdm(enumerate(range(0,360,15))):
        if kk % 4 == 0:
            fig = plt.figure(figsize=(22, 16))
            fig.tight_layout()

        ax = plt.subplot(4, 4, 4 * (kk % 4) + 1)
        dim_plot(kk, alg_name_list, prefix_list, with_op, ax)
        if kk == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(-0.6, 1))

        ax = plt.subplot(4, 4, 4 * (kk % 4) + 2)
        _2d_plot(kk, ax)

        ax = plt.subplot(4, 4, 4 * (kk % 4) + 3, projection='3d')
        _3d_plot(kk, ax)


        ax = plt.subplot(4, 4, 4 * (kk % 4) + 4)
        _1d_plot(kk, ax)

        if (kk % 4) == 3:
            path_fig = os.path.join(Consts.baseline_dir, save_file + '_{}.pdf'.format(kk // 4))
           # fig.suptitle("AVG BEST OBSERVED COMPARE - DIM {}".format(dim), fontsize=20)
            plt.savefig(path_fig)
            plt.close()

   #  path_fig = os.path.join(Consts.baseline_dir, save_file + '_{}.pdf'.format(kk // 4))
   # # fig.suptitle("AVG BEST OBSERVED COMPARE - DIM {}".format(dim), fontsize=20)
   #  plt.savefig(path_fig)
   #  plt.close()

def get_csv_from_run(optimizer, disp_name, dim):
    if dim == 784:
        res_dir = os.path.join(Consts.outdir, optimizer, 'analysis')
    else:
        res_dir = os.path.join(Consts.outdir, optimizer + '_' + str(dim), 'analysis')
    dirs = os.listdir(res_dir)

    data = defaultdict(list)
    i_num = ['01','02','03','04','05','71','72','73','74','75','76','77','78','79','80']
    f_num = list(range(1, 25 ,1))

    if dim==1:
        prefix = '1D'
        dim_c = 2
    elif dim==784:
        prefix = 'vae_vae'
        dim_c = 10
    else:
        prefix = 'coco'
        dim_c = dim

    for dir in dirs:
        try:
            index = int(dir)
            id = '{}_bbob_f{:03d}_i{}_d{:02}'.format(prefix, f_num[index // 15], i_num[index % 15], max(dim_c, 2))
            path = os.path.join(res_dir, dir)
            pi_best = np.load(os.path.join(path, 'best_list_with_explore.npy'), allow_pickle=True)
        except:
            continue
        number_of_evaluations = len(pi_best)
        best_observed = pi_best.min()
        data['id'].append(id)
        data[disp_name+'_best_observed'].append(best_observed)
        data[disp_name + '_number_of_evaluations'].append(number_of_evaluations)

    df = pd.DataFrame(data)
    save_dir = os.path.join(Consts.baseline_dir, 'results', disp_name)
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass

    file = os.path.join(save_dir, '{}_{}.csv'.format(disp_name, dim))
    df.to_csv(file, index=False)


def merge_bbo(optimizers=[], disp_name=[], dimension=[1, 2, 3, 5, 10, 20, 40, 784], save_file='baseline_cmp.pdf', plot_sum=False, need_merged=True):
    compare_file = os.path.join(Consts.baseline_dir, 'compare.csv')
    assert os.path.exists(compare_file), 'no compare file'

    baseline_df = pd.read_csv(os.path.join(compare_file))
    res_dir = os.path.join(Consts.baseline_dir, 'results')


    for i, op in enumerate(optimizers):
        op_df = pd.read_csv(os.path.join(res_dir, op, op+'_{}'.format(dimension[0]) + ".csv"))

        for dim in dimension[1:]:
            df_d = pd.read_csv(os.path.join(res_dir, op, op+'_{}'.format(dim) + ".csv"))
            op_df = op_df.append(df_d, ignore_index=True)

        if need_merged:
            op_df = op_df[['id', 'best_observed', 'number_of_evaluations']]
            op_df = op_df.rename(columns={"best_observed": disp_name[i]+"_best_observed",
                                    "number_of_evaluations": disp_name[i]+"_number_of_evaluations"})

        baseline_df = baseline_df.merge(op_df, on='id')

    columns = baseline_df.columns
    best_observed = []
    for item in columns:
        if item.startswith('trust'):
            continue
        if item.endswith('_best_observed'):
            best_observed.append(item)

    baseline_df['min_val'] = baseline_df[best_observed].astype(np.float).min(axis=1)

    file = os.path.join(Consts.baseline_dir, 'tmp_compare.csv')
    baseline_df.to_csv(file, index=False)

    X = [40*i for i in range(len(dimension))]

    fig = plt.figure(figsize=(32, 7))
    ax = plt.subplot(111)
    w = 3

    for j, best_observed_op in enumerate(best_observed):
        res = []
        x = [X[i] + j*w for i in range(len(dimension))]
        for dim in dimension:
            temp_df = baseline_df[baseline_df.dim == dim]
            if plot_sum:
                compare_method = (temp_df[best_observed_op].values - temp_df['min_val'].values) / np.abs(temp_df['f0'].values)
            else:
                compare_method_1 = (np.abs(temp_df[best_observed_op].astype(np.float).values - temp_df['min_val'].astype(np.float).values) < 1)
                #compare_method = (np.abs(temp_df[best_observed_op].astype(np.float).values - temp_df['min_val'].astype(np.float).values) < 1)
                compare_method_2 = (temp_df[best_observed_op].astype(np.float).values - temp_df['min_val'].astype(np.float).values)/temp_df['min_val'].astype(np.float).values < 0.001
                compare_method = np.bitwise_and(compare_method_1, compare_method_2)
            count = len(compare_method)
            res.append(100*compare_method.sum()/count)

        optim = best_observed_op[:-len('_best_observed')]
        if optim == 'fmin':
            ax.bar(x, res, width=w, color=optimization_colors[optim], align='center', label='Nelder Mead')
        elif optim == 'cma':
            ax.bar(x, res, width=w, color=optimization_colors[optim], align='center', label='CMA-ES')
        else:
            ax.bar(x, res, width=w, color=optimization_colors[optim], align='center', label=optim.upper())

    ax.set_xticks([i + 8 + len(dimension)//2 for i in X])
    ax.set_xticklabels(dimension, fontsize=22)
    ax.set_yticks(np.arange(0, 100, 20))
    ax.set_yticklabels(['{:.1f}'.format(i) for i in np.arange(0, 1.1, 0.2)], fontsize=22)
    ax.set_xlabel('problem dimension', fontsize=24)
    ax.set_ylabel('success rate', fontsize=24)
    ax.grid(True, axis='y')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    #ax.autoscale(tight=True)

    ax.legend(prop={'size': 24}, bbox_to_anchor=(1, 1), borderaxespad=0)
    path_fig = os.path.join(res_dir, save_file)
    plt.savefig(path_fig)
    plt.close()


def bbo_evaluate_compare(dim, index, prefix='CMP'):

    optimizer_res = get_baseline_cmp(dim, index)
    min_val = optimizer_res['min_opt'][0]
    optimizer_min_val = min_val
    f0 = optimizer_res['f0'][0]

    bbo_min_val = f0

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

    for i, key in enumerate(compare_dirs.keys()):
        path = compare_dirs[key]
        try:
            pi_best = np.load(os.path.join(path, 'best_observed.npy'), allow_pickle=True)

            bbo_min_val = min(bbo_min_val, pi_best.min())
            min_val = min(min_val, bbo_min_val)
        except:
            pass

    min_val = min_val - 0.0001
    for i, key in enumerate(compare_dirs.keys()):
        path = compare_dirs[key]
        try:
            pi_eval = np.load(os.path.join(path, 'reward_pi_evaluate.npy'), allow_pickle=True)
            frame_eval = np.load(os.path.join(path, 'frame_pi_evaluate.npy'), allow_pickle=True)
            pi_best = np.load(os.path.join(path, 'best_observed.npy'), allow_pickle=True)
            frame = np.load(os.path.join(path, 'frame.npy'), allow_pickle=True)

            ax1.loglog(frame_eval, (pi_eval - min_val)/(f0 - min_val), color=colors[i], label=key)
            ax2.loglog(frame, (pi_best - min_val) / (f0 - min_val), color=colors[i])

        except:
            pass

    fig.legend(loc='lower left', prop={'size': 6}, ncol=3)
    ax1.grid(True, which='both')
    ax2.grid(True, which='both')
    fig.suptitle("min value is {}, min bbo value is {}".format(optimizer_min_val, bbo_min_val))
    ax1.set_title('reward_pi_evaluate')
    ax2.set_title('best_observed')

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

    assert (min(x) >= -5 and max(x) <= 5), "out of range - get_best_solution dim {} problem {}".format(dim, index)
    return x, best_val



def plot_2d_first_value():
    fig, axs = plt.subplots(1,4, figsize=(32, 7))
    colors = Consts.color
    value_dir = '/data/elkayam/gan_rl/results/RUN_value_24_1_spline_2/analysis'
    first_order_dir = '/data/elkayam/gan_rl/results/RUN_first_order_24_1_spline_2/analysis'

    problems = [120, 240, 255, 270]
    for j, iter_index in enumerate(problems):
        value_path = os.path.join(value_dir, str(iter_index))
        first_order_path = os.path.join(first_order_dir, str(iter_index))
        path_dir = os.path.join(Consts.baseline_dir, 'f_eval', '2D_Contour')
        path_res = os.path.join(path_dir, '2D_index_{}.npy'.format(iter_index))
        res = np.load(path_res, allow_pickle=True).item()
        min_val = res['z'].min()
        res['z'] -= min_val - 1e-3
        max_val = res['z'].max()
        res['z'] /= (max_val + 1e-3)

        value_x = 5 * np.load(os.path.join(value_path, 'policies.npy'), allow_pickle=True)
        first_order_x = 5 * np.load(os.path.join(first_order_path, 'policies.npy'), allow_pickle=True)

        cs = axs[j].contour(res['x0'], res['x1'], np.log(res['z']), 100)

        axs[j].plot(value_x[:, 0], value_x[:, 1], '-o', color=colors[4], markersize=2, linewidth=4, label='IGL')
        axs[j].plot(first_order_x[:, 0], first_order_x[:, 1], '-o', color=colors[0], markersize=2, linewidth=4, label='EGL')

        axs[j].grid(b=True, which='both')
        axs[j].set_xticklabels([])
        axs[j].set_yticklabels([])
        axs[j].set_xlabel('({})'.format('abcd'[j]), fontsize=14)
        axs[j].axis('equal')

    axs[3].legend(prop={'size': 28}, bbox_to_anchor=(1.04, 1), borderaxespad=0)

    plt.savefig(os.path.join(Consts.baseline_dir, 'egl_2d_compare.pdf'), bbox_inches='tight')

def plot_2d_same_eps():
    fig, axs = plt.subplots(1,1, figsize=(32, 8))
    colors = Consts.color
    iter_index = '120'
    e1_path = os.path.join('/data/elkayam/gan_rl/results/CMP_first_order_ep_05_2/analysis',iter_index)
    e2_path = os.path.join('/data/elkayam/gan_rl/results/CMP_first_order_ep_2_2/analysis',iter_index)

    labels = ['eps_0.05', 'eps_0.1', 'eps_0.2']

    path_dir = os.path.join(Consts.baseline_dir, 'f_eval', '2D_Contour')
    path_res = os.path.join(path_dir, '2D_index_{}.npy'.format(iter_index))
    res = np.load(path_res, allow_pickle=True).item()
    min_val = res['z'].min()
    res['z'] -= min_val - 1e-3
    max_val = res['z'].max()
    res['z'] /= (max_val + 1e-3)

    e = []
    e.append(5 * np.load(os.path.join(e1_path, 'policies.npy'), allow_pickle=True))
    e.append(5 * np.load(os.path.join(e2_path, 'policies.npy'), allow_pickle=True))


    i=0
    for j, e_x in enumerate(e):
        cs = axs[i].contour(res['x0'], res['x1'], np.log(res['z']), 100)
        axs[i].plot(e[i][:, 0], e[i][:, 1], '-o', color=colors[i+3], markersize=2, linewidth=4, label=labels[i])

        axs[i].grid(b=True, which='both')
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        axs[i].set_xlabel(labels[i], fontsize=28)
        axs[i].axis('equal')

    #axs[0].legend(prop={'size': 14})

    plt.savefig(os.path.join(Consts.baseline_dir, 'egl_eps_compare.pdf'), bbox_inches='tight')



def plot_2d_first_divergance():
    fig, axs = plt.subplots(1,4, figsize=(32, 7))
    colors = Consts.color
    first_order_dir = '/data/elkayam/gan_rl/results/CMP_first_order_plot_2/analysis'

    problems = [201, 14, 120, 97]
    for j, iter_index in enumerate(problems):
        first_order_path = os.path.join(first_order_dir, str(iter_index))
        path_dir = os.path.join(Consts.baseline_dir, 'f_eval', '2D_Contour')
        path_res = os.path.join(path_dir, '2D_index_{}.npy'.format(iter_index))
        res = np.load(path_res, allow_pickle=True).item()
        min_val = res['z'].min()
        res['z'] -= min_val - 1e-3
        max_val = res['z'].max()
        res['z'] /= (max_val + 1e-3)

        cs = axs[j].contour(res['x0'], res['x1'], np.log(res['z']), 100)

        first_order_x = 5 * np.load(os.path.join(first_order_path, 'policies.npy'), allow_pickle=True)
        frame_policy = np.load(os.path.join(first_order_path, 'frame_pi_evaluate.npy'), allow_pickle=True)
        divergence = np.load(os.path.join(first_order_path, 'divergence.npy'), allow_pickle=True)
        frame = np.load(os.path.join(first_order_path, 'frame.npy'), allow_pickle=True)

        min = 0
        for i in set(divergence):
            if i > 10:
                break
            max = frame[divergence == i].max()
            index = np.bitwise_and(frame_policy >= min, frame_policy <= max)
            axs[j].plot(first_order_x[index, 0], first_order_x[index, 1], '-o', color=colors[(i + 1) % len(colors)], markersize=2, linewidth=4, label='tr_{}'.format(i))
            min = max + 1

        axs[j].grid(b=True, which='both')
        axs[j].set_xticklabels([])
        axs[j].set_yticklabels([])
        axs[j].set_xlabel('({})'.format('abcd'[j]), fontsize=28)
        axs[j].axis('equal')

    axs[3].legend(prop={'size': 28}, bbox_to_anchor=(1.04, 1), borderaxespad=0)

    plt.savefig(os.path.join(Consts.baseline_dir, 'egl_2d_compare_divergence.pdf'), bbox_inches='tight')


def coco_visualization():
    fig = plt.figure(figsize=(22, 16))

    for i, problem_index in tqdm(enumerate([7, 192, 223, 253])):
        ax = plt.subplot(3, 4, i+1)

        suite = cocoex.Suite("bbob", "", ("dimensions: 2"))
        problem = suite.get_problem(problem_index)
        upper_bound = problem.upper_bounds
        lower_bound = problem.lower_bounds
        interval = 0.1

        x0 = np.arange(lower_bound[0], upper_bound[0] + interval, interval)
        x1 = np.arange(lower_bound[1], upper_bound[1] + interval, interval)
        x0, x1 = np.meshgrid(x0, x1)
        z = np.zeros(x0.shape)

        for k in range(x0.shape[0]):
            for j in range(x1.shape[1]):
                x = np.array([x0[k, j], x1[k, j]])
                z[k, j] = problem(x)

        min_val = z.min()
        z -= min_val - 1e-3
        max_val = z.max()
        z /= (max_val + 1e-3)

        ax.contour(x0, x1, np.log(z), 100)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('({}1)'.format('abcd'[i]), fontsize=16)
        ax.axis('equal')

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

        ax = plt.subplot(3, 4, 5+i, projection='3d')

        ax.plot_trisurf(x_list[:, 0], x_list[:, 1], f_list, cmap='winter')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.set_xlabel('({}2)'.format('abcd'[i]), fontsize=16)

        interval = 0.0001
        x0 = np.arange(-1, 1 + interval, interval)
        norm_policy = np.clip(one_d_change_dim(x0), -1, 1)
        f = np.zeros(x0.shape)
        policy = 0.5 * (norm_policy + 1) * (upper_bound - lower_bound) + lower_bound

        for k in range(x0.shape[0]):
            f[k] = problem(policy[k])

        ax = plt.subplot(3, 4, 9+i)

        ax.plot(policy, f, color='g', markersize=2, linewidth=4, label='f')
        ax.grid(True, which='both')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('({}3)'.format('abcd'[i]), fontsize=16)


    plt.savefig(os.path.join(Consts.baseline_dir, 'egl_coco_visualization.pdf'), bbox_inches='tight')



def plot_2d_first_value_and_divergance():
    fig, axs = plt.subplots(1,4, figsize=(32, 7))
    colors = Consts.color

    #################################################
    problems = [194, 97]
    for k, iter_index in enumerate(problems):
        j = k

        e1_path = os.path.join('/data/elkayam/gan_rl/results/CMP_first_order_ep_05_2/analysis', str(iter_index))
        e2_path = os.path.join('/data/elkayam/gan_rl/results/CMP_first_order_ep_2_2/analysis', str(iter_index))

        labels = [r'$\varepsilon=0.05$', r'$\varepsilon=0.2$']

        path_dir = os.path.join(Consts.baseline_dir, 'f_eval', '2D_Contour')
        path_res = os.path.join(path_dir, '2D_index_{}.npy'.format(iter_index))

        res = np.load(path_res, allow_pickle=True).item()
        min_val = res['z'].min()
        res['z'] -= min_val - 1e-3
        max_val = res['z'].max()
        res['z'] /= (max_val + 1e-3)

        e = []
        e.append(5 * np.load(os.path.join(e1_path, 'policies.npy'), allow_pickle=True))
        e.append(5 * np.load(os.path.join(e2_path, 'policies.npy'), allow_pickle=True))

        for i, e_x in enumerate(e):
            cs = axs[j].contour(res['x0'], res['x1'], np.log(res['z']), 100)
            axs[j].plot(e[i][:, 0], e[i][:, 1], '-o', color=colors[(i+7)], markersize=2, linewidth=4, label=labels[i])

        axs[j].grid(b=True, which='both')
        axs[j].set_xticklabels([])
        axs[j].set_yticklabels([])
        axs[j].set_xlabel('({})'.format('abcd'[j]), fontsize=28)
        axs[j].axis('equal')

    axs[0].legend(prop={'size': 28}, loc='upper left', borderaxespad=0)
    #################################################



    value_dir = '/data/elkayam/gan_rl/results/RUN_value_24_1_spline_2/analysis'
    first_order_dir = '/data/elkayam/gan_rl/results/RUN_first_order_24_1_spline_2/analysis'

    problems = [255, 240]
    for k, iter_index in enumerate(problems):
        j = k + 2
        value_path = os.path.join(value_dir, str(iter_index))
        first_order_path = os.path.join(first_order_dir, str(iter_index))
        path_dir = os.path.join(Consts.baseline_dir, 'f_eval', '2D_Contour')
        path_res = os.path.join(path_dir, '2D_index_{}.npy'.format(iter_index))
        res = np.load(path_res, allow_pickle=True).item()
        min_val = res['z'].min()
        res['z'] -= min_val - 1e-3
        max_val = res['z'].max()
        res['z'] /= (max_val + 1e-3)

        value_x = 5 * np.load(os.path.join(value_path, 'policies.npy'), allow_pickle=True)
        first_order_x = 5 * np.load(os.path.join(first_order_path, 'policies.npy'), allow_pickle=True)

        cs = axs[j].contour(res['x0'], res['x1'], np.log(res['z']), 100)

        axs[j].plot(value_x[:, 0], value_x[:, 1], '-o', color=colors[4], markersize=2, linewidth=4, label='IGL')
        axs[j].plot(first_order_x[:, 0], first_order_x[:, 1], '-o', color=colors[0], markersize=2, linewidth=4, label='EGL')

        axs[j].grid(b=True, which='both')
        axs[j].set_xticklabels([])
        axs[j].set_yticklabels([])
        axs[j].set_xlabel('({})'.format('abcd'[j]), fontsize=28)
        axs[j].axis('equal')

    axs[3].legend(prop={'size': 28}, loc='lower left')

    plt.savefig(os.path.join(Consts.baseline_dir, 'egl_2d_compare_value_eps.pdf'), bbox_inches='tight')

def plot_avg_dim():
    fig, axs = plt.subplots(1, 4, figsize=(32, 7))

    prefix = ['RUN_value_24_1_spline', 'RUN_first_order_24_1_spline']
    alg_name = ['IGL', 'EGL']

    #dims = [10, 40, 784]
    #dims = [40, 784]
    dims = [40]
    y_label = True
    for i, dim in enumerate(dims):
        if dim == 784:
            prefix = ['RUN_value_24_1_784', 'RUN_first_order_24_1_784']

        axs[i] = avg_dim_best_observed(dim=dim, save_file='', alg_name_list=alg_name, prefix_list=prefix, with_op=True, axs=axs[i], y_label=y_label)

        axs[i].set_xlabel('t\n({})'.format('abcd'[i]), fontsize=24)
        axs[i].axis('tight')
        y_label = False
    #axs[0].legend(prop={'size': 20}, bbox_to_anchor=(-0.25, 1), borderaxespad=0)
    axs[0].legend(prop={'size': 18}, loc='lower left')
    axs[0].set_xlim(left=1, right=150000)
    axs[0].set_ylim(bottom=2**(-7), top=1.5)
    #axs[0].legend(prop={'size': 14})



    dim = 784
    axs_i=2
    prefix = ['RUN_first_order_24_1', 'ABL_EGL_fc_tr_map_800_784', 'RUN_value_24_1', 'ABL_IGL_fc_tr_map_800_784']
    alg_name = ['EGL_64', 'EGL_800', 'IGL_64', 'IGL_800']
    # prefix = ['RUN_first_order_24_1', 'ABL_first_order_fc_tr_map_256',
    #           'RUN_value_24_1', 'ABL_IGL_256']
    #alg_name = ['EGL_64', 'EGL_256', 'IGL_64', 'IGL_256']
    avg_dim_best_observed(dim=dim, save_file='', alg_name_list=alg_name, prefix_list=prefix, with_op=True, axs=axs[axs_i])
    axs[axs_i].set_xlabel('t\n({})'.format('abcd'[axs_i]), fontsize=24)
    axs[axs_i].legend(prop={'size': 18}, loc='lower left')
    axs[axs_i].set_xlim(left=384, right=150000)
    axs[axs_i].set_ylim(bottom=2 ** (-7), top=1.5)

    dim = 40
    axs_i = 3
    prefix = ['ABL_EGL_r1_pertub_0_40', 'ABL_EGL_r1_pertub_1_40',
              'ABL_EGL_pertub_2_40',
              'ABL_EGL_r4_pertub_0_40', 'ABL_EGL_r16_pertub_0']
    div_name = 'RB1_P0'
    alg_name = ['RB1_P0', 'RB1_P1e-1',
                'RB1_P1e-2',
                'RB4_P0', 'RB16_P0']
    avg_perturb_best_observed(dim=dim, save_file='', div_name=div_name, alg_name_list=alg_name, prefix_list=prefix, axs=axs[axs_i])
    axs[axs_i].set_xlabel('t\n({})'.format('abcd'[axs_i]), fontsize=24)
    axs[axs_i].legend(prop={'size': 18}, loc='lower left')

    dim = 40
    axs_i = 1
    # prefix = ['ABL_first_order_fc_no_tr_no_map', 'ABL_first_order_fc_no_tr_map',
    #           'ABL_first_order_fc_tr_no_map',
    #           'ABL_first_order_fc_tr_map'
    #           # ,'ABL_first_order_spline_mor_tr_map'
    #           ]
    # alg_name = ['FC', 'FC_OM', 'FC_TR', 'FC_TR_OM'
    #     # , 'SPLINE'
    #             ]

    prefix = ['ABL_EGL_fc_no_tr_no_map_40',
              'ABL_EGL_fc_no_tr_map_40',
              'ABL_EGL_fc_tr_no_map_40',
              'ABL_EGL_fc_tr_map_40',
              'ABL_EGL_spline_tr_map_40']
    alg_name = ['FC', 'FC_OM', 'FC_TR', 'FC_TR_OM', 'SPLINE']

    avg_dim_best_observed(dim=dim, save_file='', alg_name_list=alg_name, prefix_list=prefix, with_op=False, axs=axs[axs_i])
    axs[axs_i].set_xlabel('t\n({})'.format('abcd'[axs_i]), fontsize=24)
    axs[axs_i].legend(prop={'size': 18}, loc='lower left')
    axs[axs_i].set_xlim(left=384, right=150000)
    axs[axs_i].set_ylim(bottom=2 ** (-7), top=1.5)

    fig.savefig(os.path.join(Consts.baseline_dir, 'egl_dim_avg_spline.pdf'), bbox_inches='tight')

if __name__ == '__main__':

    # for dim in [2, 3, 5, 10, 20, 40]:
    #     get_csv_from_run('RUN_first_order_24_1_spline', 'EGL', dim)
    #     get_csv_from_run('RUN_value_24_1_spline', 'IGL', dim)
    #
    # for dim in [784]:
    #     get_csv_from_run('RUN_first_order_24_1_784', 'EGL', dim)
    #     get_csv_from_run('RUN_value_24_1_784', 'IGL', dim)
    #
    # optimizers = ['IGL', 'EGL']
    # disp_name = ['IGL', 'EGL']
    # dims = [2, 3, 5, 10, 20, 40, 784]
    #
    # merge_bbo(optimizers=optimizers, disp_name=disp_name, dimension=dims, save_file='egl_baseline_cmp_success.pdf', plot_sum=False, need_merged=False)

    plot_avg_dim()

    # prefix = ['RUN_value_24_1_spline', 'RUN_first_order_24_1_spline']
    # alg_name = ['IGL', 'EGL']
    #
    # # dims = [10, 40, 784]
    # # dims = [40, 784]
    # dims = [40]
    # for i, dim in enumerate(dims):
    #     avg_dim_best_observed(dim=dim, save_file='40.pdf', alg_name_list=alg_name, prefix_list=prefix, with_op=True)
    #
    #
    # dim = 784
    # prefix = ['RUN_first_order_24_1', 'ABL_EGL_fc_tr_map_800_784', 'RUN_value_24_1', 'ABL_IGL_fc_tr_map_800_784']
    # alg_name = ['EGL_64', 'EGL_800', 'IGL_64', 'IGL_800']
    # avg_dim_best_observed(dim=dim, save_file='CMP_VAE.pdf', alg_name_list=alg_name, prefix_list=prefix, with_op=True)


    dim = 40
    prefix = ['RUN_value_24_1_spline', 'RUN_first_order_24_1_spline']
    alg_name = ['IGL', 'EGL']
    # avg_dim_best_observed(dim=dim, save_file='my40.pdf', alg_name_list=alg_name, prefix_list=prefix, with_op=True)


    dim = 40
    axs_i = 1
    prefix = ['ABL_EGL_fc_no_tr_no_map_40',
              'ABL_EGL_fc_no_tr_map_40',
              'ABL_EGL_fc_tr_no_map_40',
              'ABL_EGL_fc_tr_map_40',
              'ABL_EGL_spline_tr_map_40']
    alg_name = ['FC', 'FC_OM', 'FC_TR', 'FC_TR_OM', 'SPLINE']
    # avg_dim_best_observed(dim=dim, save_file='NEW_TR_OM.pdf', alg_name_list=alg_name, prefix_list=prefix, with_op=False)

    #

    # prefix_list = ['RUN_value_24_1_spline', 'RUN_first_order_24_1_spline']
    # alg_name_list = ['IGL', 'EGL']
    # dims = [2]
    # for dim in dims:
    #     save_file = '{} dim {} avg_dim_problem'.format('CMP', dim)
    #     avg_dim_problem_2d(dim, save_file, alg_name_list, prefix_list, with_op=True)

    #
    # dims = [40]
    # for dim in dims:
    #     save_file = '{} dim {} avg_dim_problem'.format('CMP', dim)
    #     avg_dim_problem(dim, save_file, alg_name_list, prefix_list, with_op=True)
    #
    # dim=784
    # prefix = ['RUN_value_24_1_784', 'RUN_first_order_24_1_784']
    # alg_name_list = ['IGL', 'EGL']
    # save_file = '{} dim {} avg_dim_problem'.format('CMP', dim)
    # avg_dim_problem(dim, save_file, alg_name_list, prefix_list, with_op=True)


    #plot_2d_same_eps()
    # plot_2d_first_value_and_divergance()
    # #plot_2d_first_value()
    # #plot_2d_first_divergance()


    # coco_visualization()

    # plot_avg_dim()

    #merge_baseline_one_line_compare(dims=[784, 1, 2, 3, 5, 10, 20, 40])
    # # #
    # # # #
    # # #optimizers = ['first_order_clip0', 'first_order_clip1', 'first_order_clip1_cone1']
    # # #merge_bbo(optimizers=optimizers, disp_name=disp_name, dimension=dims, save_file='baseline_cmp_avg_sum.pdf', plot_sum=True)
    # # #
    #bbo_evaluate_compare(dim=784, index=2, prefix='RUN')
    # # # #
    # # #
    # #dims = [1,2,3,5,10,20,40]


    # #visualization(120)
    #
    # # # for i in tqdm(range(0, 360, 1)):
    # # #     visualization(i)
    # #
    # # #treeD_plot_contour(0)
    # # #calc_f0()
    # # #twoD_plot_contour(index)
    # #


    # filter_mod = 15
    # dim = 784
    # for i in tqdm(range(79, 83, 1)):
    #     compare_problem_baseline(dim, i, budget=150000)
