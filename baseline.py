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
from matplotlib import cm

username = pwd.getpwuid(os.geteuid()).pw_name

if username == 'morsi':
    from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!
    base_dir = os.path.join('/Users', username, 'Desktop', 'baseline')
else:
    from vae import VaeProblem, VAE
    base_dir = os.path.join('/data/', username, 'gan_rl', 'baseline')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

color = ['b', 'g', 'r', 'y', 'c', 'm', 'k', 'lime', 'gold', 'slategray', 'indigo', 'maroon', 'plum', 'pink', 'tan', 'khaki', 'silver',
             'navy', 'skyblue', 'teal', 'darkkhaki', 'indianred', 'orchid', 'lightgrey', 'dimgrey']

epsilon = 10

def merge_baseline(dims=[2, 3, 5, 10, 20, 40, 784]):

    data = defaultdict(list)
    min_data = defaultdict(list)
    for dim in dims:
        for index in range(360):
            optimizer_res = get_baseline_cmp(dim, index)
            if dim == 784:
                min_res_cmp = get_baseline_cmp(20, index)
            else:
                min_res_cmp = get_baseline_cmp(dim, index)
            min_val = float(min(min_res_cmp.best_observed))
            min_data['dim'].append(dim)
            min_data['iter_index'].append(index)
            min_data['min_val'].append(min_val)
            for op in optimizer_res['fmin']:
                res = optimizer_res[optimizer_res['fmin'] == op]
                best_observed = float(res['best_observed'])
                data['dim'].append(dim)
                data['iter_index'].append(index)
                data['optimizer'].append(op)
                data['budget'].append(float(res.number_of_evaluations))
                if np.abs(best_observed - min_val) < epsilon:
                    data['success'].append(1)
                else:
                    data['success'].append(0)

    df = pd.DataFrame(data)
    file = os.path.join(base_dir, 'success.csv')
    df.to_csv(file)
    df = pd.DataFrame(min_data)
    file = os.path.join(base_dir, 'min_val.csv')
    df.to_csv(file)

def merge_baseline_mor(dims=[2, 3, 5, 10, 20, 40, 784]):

    data = defaultdict(list)
    for dim in dims:
        for index in range(360):
            optimizer_res = get_baseline_cmp(dim, index)

            data['dim'].append(dim)
            data['iter_index'].append(index)

            for i, op in enumerate(optimizer_res['fmin']):
                res = optimizer_res[optimizer_res['fmin'] == op]
                data[op + '_best_observed'].append(float(res.best_observed))
                data[op + '_budget'].append(float(res.number_of_evaluations))
                data[op + '_x'].append(res.x)


    df = pd.DataFrame(data)
    file = os.path.join(base_dir, 'compare.csv')
    df.to_csv(file)

def plot_res(optimizers = [], max_budget = 1200, compare_baseline=False):
    res_dir = os.path.join(base_dir, 'results')
    dimension = [2, 3, 5, 10, 20, 40, 784]
    success_df = pd.read_csv(os.path.join(base_dir, "success.csv"))
    min_df = pd.read_csv(os.path.join(base_dir, 'min_val.csv'))
    if compare_baseline:
        cmp_optim = ['fmin_slsqp', 'fmin', 'fmin_cobyla', 'fmin_powell', 'fmin_cg', 'fmin_bfgs', 'fmin2']
    else:
        cmp_optim = []

    index = set(range(360))
    for dim in dimension:
        for op in optimizers:
            tmp = pd.read_csv(os.path.join(res_dir, op, op+'_{}'.format(dim) + ".csv"))
            index = index.intersection(tmp['iter_index'])
            tmp = tmp.loc[tmp.iter_index.isin(index)]
            tmp_min_df = min_df.loc[(min_df.dim == dim) & (min_df['iter_index'].isin(index))]
            success = np.abs(np.min([tmp['best_observed'].values, tmp_min_df['min_val'].values], axis=0) - tmp['best_observed'].values) < epsilon
            op_df = pd.DataFrame({'dim': [dim]*len(index), 'iter_index': list(index), 'optimizer':[op]*len(index),
                                  'success': list(success.astype('int')), 'budget': [max_budget]*len(index)})
            success_df = success_df.append(op_df, ignore_index=True, sort=False)

    cmp_size = len(index)
    if cmp_size == 0:
        return

    success_df = success_df[success_df['iter_index'].isin(index)]
    X = [20*i for i in range(len(dimension))]
    ax = plt.subplot(111)
    w = 1

    opl_opt = cmp_optim + optimizers
    for j, op in enumerate(opl_opt):
        x = [X[i] + j*w for i in range(len(dimension))]
        res = [success_df[(success_df['optimizer'] == op) & (success_df['dim'] == dim) & (success_df['budget'] <= max_budget)].success.sum()/cmp_size for dim in dimension]
        ax.bar(x, res, width=w, color=color[j], align='center', label=op)

    ax.set_xticks([i + len(dimension)//2 for i in X])
    ax.set_xticklabels(dimension)
   # ax.autoscale(tight=True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
    #ax.legend()
    plt.title("BASELINE COMPARE - BUDGET {}".format(max_budget))
    plt.show()


def get_baseline_cmp(dim, index):
    optimizer_res = pd.read_csv(os.path.join(base_dir, 'compare', 'dim_{} index_{}'.format(dim, index) + ".csv"))
    return optimizer_res


def get_min_val(dim, index):
    min_df = pd.read_csv(os.path.join(base_dir, 'min_val.csv'))
    return float(min_df[(min_df.dim == dim) & (min_df.iter_index == index)].min_val)

def compare_beta_evaluate(dim, index, path, title, baseline_cmp = False):

    min_val = get_min_val(dim, index)
    compare_file = 'beta_evaluate.npy'#'best_observed.npy' #beta_evaluate
    analysis_path = os.path.join(path)
    dirs = os.listdir(analysis_path)
    compare_dict = {}

    my_optim = []
    for dir in dirs:
        try:
            my_optim.append(dir)
            path = os.path.join(analysis_path, dir, str(index))
            files = os.listdir(path)
            if compare_file in files:
                f_val = np.load(os.path.join(path, compare_file))
                compare_dict[dir] = [np.arange(f_val.size), f_val]
        except:
            continue

    if baseline_cmp:
        baseline_df = get_baseline_cmp(dim, index)
        for i in range(len(baseline_df)):
            f_val = np.array([float(i) for i in baseline_df.iloc[i].f[1:-1].split(',')])
            e_val = np.array([float(i) for i in baseline_df.iloc[i].eval[1:-1].split(',')])
            compare_dict[baseline_df.iloc[i].fmin] = [e_val, f_val]

    res_keys = compare_dict.keys()
    if len(res_keys) is 0:
        return

    plt.subplot(111)

    for i, key in enumerate(res_keys):
        if key in my_optim:
        #plt.plot(t, v.cumsum()/t, color=color[i], label=key)
            plt.plot(compare_dict[key][0], compare_dict[key][1] - min_val, color=color[i], label=key)
        else:
            plt.plot(compare_dict[key][0], compare_dict[key][1] - min_val, markerfacecolor='None', marker='x', color=color[i], label=key)

    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.legend()
    plt.title(title)
   # plt.ylim([0, 200])
    plt.xlim([0, 200])
    plt.show()

def plot_2D(problem_index, path, save_fig=False):

    path_dir = os.path.join(base_dir, '2D')
    path_res = os.path.join(path_dir, '2D_index_{}.npy'.format(problem_index))
    res = np.load(path_res)

    x = np.load(os.path.join(path, str(problem_index), 'policies.npy'))
    x_exp = np.load(os.path.join(path, str(problem_index), 'explore_policies.npy')).reshape(-1, 2)

    if dim != 784:
        x *= 5
        x_exp *= 5

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #mean = res[:, 2].mean()
    # std = res[:, 2].std()
    # res[:, 2] = (res[:, 2] - mean) / std
    ax.plot_trisurf(res[:, 0], res[:, 1], res[:, 2], cmap='winter')
    plt.plot(x[:, 0], x[:, 1], '-o', color='b', markersize=1)
    plt.plot(x_exp[:, 0], x_exp[:, 1], '.', color='r', markersize=1)

    ax.set_title('2D_index_{}'.format(problem_index))
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('f(x0,x1)')

    if save_fig:
        path_dir_fig = os.path.join(path_dir, 'figures')
        if not os.path.exists(path_dir_fig):
            os.makedirs(path_dir_fig)

        path_fig = os.path.join(path_dir_fig, '2D_index_{}.pdf'.format(problem_index))
        plt.savefig(path_fig)
    else:
        plt.show()

    plt.close()

def plot_2D_contour(problem_index, path, save_fig=False):

    path_dir = os.path.join(base_dir, '2D_temp')
    path_res = os.path.join(path_dir, '2D_index_{}.npy'.format(problem_index))
    res = np.load(path_res).item()

    x = np.load(os.path.join(path, str(problem_index), 'policies.npy'))
    x_exp = np.load(os.path.join(path, str(problem_index), 'explore_policies.npy')).reshape(-1,2)

    if dim != 784:
        x *= 5
        x_exp *= 5
    #f = np.load(os.path.join(path, str(problem_index), 'beta_evaluate.npy'))


    fig, ax = plt.subplots()
    cs = ax.contour(res['x0'], res['x1'], res['z'], 100)
    plt.plot(x_exp[:, 0], x_exp[:, 1], '.', color='r', markersize=1)
    plt.plot(x[:, 0], x[:, 1], '-o', color='b', markersize=1)
    plt.title(path.split('/')[-1])
    fig.colorbar(cs)
    if save_fig:
        path_dir_fig = os.path.join(path_dir, 'figures')
        if not os.path.exists(path_dir_fig):
            os.makedirs(path_dir_fig)

        path_fig = os.path.join(path_dir_fig, '2D_index_{}.pdf'.format(problem_index))
        plt.savefig(path_fig)
    else:
        plt.show()


    plt.close()


if __name__ == '__main__':
    #merge_baseline()

    #merge_baseline_mor()

    dim = 2
    index = 15
    dir_name = 'LR'


    path = os.path.join(base_dir, 'analysis', dir_name, str(dim))

    title = "{} dim = {} index = {}".format(dir_name, dim, index)
    compare_beta_evaluate(dim, index, path, title, baseline_cmp=False)


    #plot_res(optimizers=["bbo", "grad"], max_budget=12000, compare_baseline=True)

    # for i in range(360):
    #     plot_2D(i, save_fig=True)

    prefix = 'lr_3_debug'
    path = os.path.join(base_dir, 'analysis', dir_name, str(dim), prefix)
    #plot_2D_contour(index, path, False)
    #plot_2D(15, path, False)
