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
import matplotlib.pyplot as plt
from collections import defaultdict
from vae import VaeProblem, VAE
from environment import EnvCoco, EnvOneD, EnvVae
from environment import one_d_change_dim
import pickle
username = pwd.getpwuid(os.geteuid()).pw_name

if username == 'morsi':
    base_dir = os.path.join('/Users', username, 'Desktop')
else:
    from vae import VaeProblem, VAE
    base_dir = os.path.join('/data/', username, 'gan_rl')




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
            env = EnvVae(problem)
        elif dim == 1:
            suite.reset()
            problem = suite.get_problem(index)
            env = EnvOneD(problem, False)
        else:
            suite.reset()
            problem = suite.get_problem(index)
            env = EnvCoco(problem, False)

        func = env.f

        x = env.initial_solution
        f = []
        eval = []
        try:
            for curr_budget in range(sub_budget, budget, sub_budget):
                env.limit_budget(curr_budget)

                x = run_problem(fmin, func, x, sub_budget)

                env.limit_budget(float('inf'))
                f.append(func(x))
                eval.append(problem.evaluations)
                if problem.final_target_hit or problem.evaluations > budget:
                    break
        except:
            continue

        data['fmin'].append(fmin.__name__)
        data['index'].append(problem.index)
        data['hit'].append(problem.final_target_hit)
        data['x'].append(x)
        data['id'].append(env.get_problem_id())
        data['best_observed'].append(problem.best_observed_fvalue1)
        data['number_of_evaluations'].append(problem.evaluations)
        data['eval'].append(eval)
        data['f'].append(f)
        data['f0'].append(func(env.initial_solution))

        # print(data['fmin'][-1] + ":\tevaluations: " + str(data['number_of_evaluations'][-1]) + "\tf(x): "
        #                                                   + str(func(x)) + "\thit: " + str(data['hit'][-1]))

    df = pd.DataFrame(data)
    title = 'dim_{} index_{}'.format(dim, index)
    save_dir = os.path.join(base_dir, 'baseline', 'compare')
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except:
            pass
    fmin_file = os.path.join(save_dir, title + '.csv')
    df.to_csv(fmin_file)

def run_problem(fmin,  problem, x0, budget):

        fmin_name = fmin.__name__

        if fmin_name is 'fmin_slsqp':
            x, best_val, _, _, _ = fmin(problem, x0, iter=budget, full_output=True, iprint=-1)

        elif fmin_name is 'fmin':
            x, best_val, _, eval_num, _ = fmin(problem, x0, maxfun=budget, disp=False, full_output=True)

        elif fmin_name is 'fmin2':
            x, _ = fmin(problem, x0, 2, {'maxfevals': budget, 'verbose':-9}, restarts=9)

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

    res_dir = os.path.join(base_dir, 'results')
    local_path = os.path.join('/Users', 'morsi', 'Desktop', 'baseline', 'analysis')
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
    res_dir = os.path.join(base_dir, 'baseline', '2D')
    if not os.path.exists(res_dir):
        try:
            os.makedirs(res_dir)
        except:
            pass
    path_res = os.path.join(res_dir, '2D_index_{}.npy'.format(problem_index))

    np.save(path_res, res)

def calc_f0():
    save_dir = os.path.join(base_dir, 'baseline')
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

    res_dir = os.path.join(base_dir, 'baseline', '2D_Contour')
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
    interval = 0.0001

    x0 = np.arange(-1, 1 + interval, interval)
    norm_policy = np.clip(one_d_change_dim(x0), -1, 1)
    f = np.zeros(x0.shape)
    policy = 0.5 * (norm_policy + 1) * (upper_bound - lower_bound) + lower_bound

    for i in range(x0.shape[0]):
        f[i] = problem(policy[i])

    res_dir = os.path.join(base_dir, 'baseline', '1D')
    if not os.path.exists(res_dir):
        try:
            os.makedirs(res_dir)
        except:
            pass

    path_res = os.path.join(res_dir, '1D_index_{}.pkl'.format(problem_index))

    with open(path_res, 'wb') as handle:
        pickle.dump({'norm_policy':norm_policy, 'policy':policy, 'f':f}, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_baseline_cmp(dim, index):
    optimizer_res = pd.read_csv(os.path.join(base_dir, 'baseline', 'compare', 'dim_{} index_{}'.format(dim, index) + ".csv"))
    return optimizer_res

def merge_baseline_one_line_compare(dims=[1, 2, 3, 5, 10, 20, 40, 784]):

    data = defaultdict(list)
    for dim in dims:
        for index in range(360):
            optimizer_res = get_baseline_cmp(dim, index)

            data['dim'].append(dim)
            data['iter_index'].append(index)
            data['f0'].append(max(optimizer_res.f0))
            data['id'].append(optimizer_res.id[0])
            for i, op in enumerate(optimizer_res['fmin']):
                res = optimizer_res[optimizer_res['fmin'] == op]
                data[op + '_best_observed'].append(float(res.best_observed))
                data[op + '_budget'].append(float(res.number_of_evaluations))
                data[op + '_x'].append(res.x)

    df = pd.DataFrame(data)
    file = os.path.join(base_dir, 'compare.csv')
    df.to_csv(file)

def run_baseline(dims=[1, 2, 3, 5, 10, 20, 40, 784]):
    filter_mod = 1

    for dim in dims:
        for i in tqdm(range(0, 360, filter_mod)):
            compare_problem_baseline(dim, i, budget=200)

    merge_baseline_one_line_compare(dims)

if __name__ == '__main__':
    run_baseline()
    #compare_problem_baseline(784,15,90)
    # filter_mod = 100
    #
    # for dim in ['1','2','3','5','10','20','40','784']:
    #     for i in tqdm(range(0, 360, filter_mod)):
    #         compare_problem_baseline(dim, i, budget=12000)

    # for i in tqdm(range(0, 360, 1)):
    #      D1_plot(i)
    #
    #      treeD_plot_contour(i)  #treeD_plot

    #create_copy_file("CMP", 2, 0)


    #treeD_plot_contour(0)
    #calc_f0()
    #twoD_plot_contour(index)
