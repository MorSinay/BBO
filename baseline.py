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
import pickle

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

problem_index_type_list = [1,2,3,4,5,71,72,73,74,75,76,77,78,79,80]
epsilon = 1

# def merge_baseline(dims=[1, 2, 3, 5, 10, 20, 40, 784]):
#
#     data = defaultdict(list)
#     min_data = defaultdict(list)
#     for dim in dims:
#         for index in range(360):
#             optimizer_res = get_baseline_cmp(dim, index)
#             if dim == 784:
#                 min_res_cmp = get_baseline_cmp(20, index)
#             else:
#                 min_res_cmp = get_baseline_cmp(dim, index)
#             min_val = float(min(min_res_cmp.best_observed))
#             min_data['dim'].append(dim)
#             min_data['iter_index'].append(index)
#             min_data['min_val'].append(min_val)
#             for op in optimizer_res['fmin']:
#                 res = optimizer_res[optimizer_res['fmin'] == op]
#                 best_observed = float(res['best_observed'])
#                 data['dim'].append(dim)
#                 data['iter_index'].append(index)
#                 data['optimizer'].append(op)
#                 data['budget'].append(float(res.number_of_evaluations))
#                 if np.abs(best_observed - min_val) < epsilon:
#                     data['success'].append(1)
#                 else:
#                     data['success'].append(0)
#
#     df = pd.DataFrame(data)
#     file = os.path.join(base_dir, 'success.csv')
#     df.to_csv(file)
#     df = pd.DataFrame(min_data)
#     file = os.path.join(base_dir, 'min_val.csv')
#     df.to_csv(file)

def merge_baseline_one_line_compare(dims=[2, 3, 5, 10, 20, 40, 784]):

    data = defaultdict(list)
    for dim in dims:
        for index in range(360):
            optimizer_res = get_baseline_cmp(dim, index)

            data['dim'].append(dim)
            data['iter_index'].append(index)
            data['f0'].append(max(optimizer_res.f0))
            if dim == 784:
                data['id'].append('vae_'+optimizer_res.id[0])
            else:
                data['id'].append(optimizer_res.id[0])
            for i, op in enumerate(optimizer_res['fmin']):
                res = optimizer_res[optimizer_res['fmin'] == op]
                data[op + '_best_observed'].append(float(res.best_observed))
                data[op + '_budget'].append(float(res.number_of_evaluations))
                data[op + '_x'].append(res.x)


    df = pd.DataFrame(data)
    file = os.path.join(base_dir, 'compare.csv')
    df.to_csv(file)

def merge_bbo(optimizers = [], dimension = [2, 3, 5, 10, 20, 40, 784], plot_sum=False):
    compare_file = os.path.join(base_dir, 'compare.csv')
    assert os.path.exists(compare_file), 'no compare file'

    baseline_df = pd.read_csv(os.path.join(compare_file))
    res_dir = os.path.join(base_dir, 'results')

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

    file = os.path.join(base_dir, 'tmp_compare.csv')
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
                compare_method = (temp_df[best_observed_op].values - temp_df['min_val'].values) / temp_df['f0'].values
            else:
                compare_method = (temp_df[best_observed_op].values - temp_df['min_val'].values) < epsilon
            count = len(compare_method)
            res.append(compare_method.sum()/count)
        optim = best_observed_op[:-len('_best_observed')]
        ax.bar(x, res, width=w, color=color[j], align='center', label=optim)

    ax.set_xticks([i + len(dimension)//2 for i in X])
    ax.set_xticklabels(dimension)
   # ax.autoscale(tight=True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=4)
    #ax.legend()
    #plt.title("BASELINE COMPARE - BUDGET {}".format(max_budget))
    plt.show()


def get_baseline_cmp(dim, index):
    optimizer_res = pd.read_csv(os.path.join(base_dir, 'compare', 'D_{}'.format(dim), 'dim_{} index_{}'.format(dim, index) + ".csv"))
    return optimizer_res


def get_min_f0_val(dim, index):
    min_df = pd.read_csv(os.path.join(base_dir, 'compare.csv'))
    tmp_df = min_df[(min_df.dim == dim) & (min_df.iter_index == index)]

    min_val = float(tmp_df.min_val)
    f0 = float(tmp_df.f0)
    return min_val, f0

def compare_pi_evaluate(dim, index, path, title, baseline_cmp = False):

    min_val, f0 = get_min_f0_val(dim, index)
    compare_file = 'pi_evaluate.npy'#'best_observed.npy' #pi_evaluate
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
            plt.loglog(compare_dict[key][0], (compare_dict[key][1] - min_val)/(f0 - min_val), color=color[i], label=key)
        else:
            plt.loglog(compare_dict[key][0], (compare_dict[key][1] - min_val)/(f0 - min_val), markerfacecolor='None', marker='x', color=color[i], label=key)

    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.legend()
    plt.title(title)
    plt.grid(True, which='both')
    #plt.ylim([0, 100000])
    #plt.xlim([350, 400])
    plt.show()

def plot_1D(problem_index, save_fig=False):

    path_dir = os.path.join(base_dir, 'f_eval', '1D')
    path_res = os.path.join(path_dir, '1D_index_{}.pkl'.format(problem_index))
#    res = np.load(path_res)

    with open(path_res, 'rb') as handle:
        res = pickle.load(handle)

        plt.subplot(111)

        plt.plot(res['policy'][:,0], res['f'], color='g', markersize=1, label='f')

        problem_f_type = problem_index // 15 + 1
        problem_index_type = problem_index_type_list[problem_index % 15]

        plt.title('1D_index_{} -- f_{:2d} id_{:2d}'.format(problem_index, problem_f_type, problem_index_type))
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True, which='both')

        if save_fig:
            path_dir_fig = os.path.join(path_dir, 'figures')
            if not os.path.exists(path_dir_fig):
                os.makedirs(path_dir_fig)

            path_fig = os.path.join(path_dir_fig, '1D_index_{:3d}.pdf'.format(problem_index))
            plt.savefig(path_fig)
        else:
            plt.show()

        plt.close()

def plot_2D(problem_index, path, save_fig=False):

    path_dir = os.path.join(base_dir, 'f_eval', '2D')
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

    problem_f_type = problem_index // 15 + 1
    problem_index_type = problem_index_type_list[problem_index % 15]

    ax.set_title('2D_index_{} -- f_{:2d} id_{:2d}'.format(problem_index, problem_f_type, problem_index_type))
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('f(x0,x1)')

    if save_fig:
        path_dir_fig = os.path.join(path_dir, 'figures')
        if not os.path.exists(path_dir_fig):
            os.makedirs(path_dir_fig)

        path_fig = os.path.join(path_dir_fig, '2D_index_{:3d}.pdf'.format(problem_index))
        plt.savefig(path_fig)
    else:
        plt.show()

    plt.close()

def plot_2D_contour(problem_index, path, save_fig=False):

    path_dir = os.path.join(base_dir, 'f_eval', '2D_Contour')
    path_res = os.path.join(path_dir, '2D_index_{}.npy'.format(problem_index))
    res = np.load(path_res).item()

    x = np.load(os.path.join(path, str(problem_index), 'policies.npy'))
    x_exp = np.load(os.path.join(path, str(problem_index), 'explore_policies.npy')).reshape(-1,2)[120:]

    fig, ax = plt.subplots()
    cs = ax.contour(res['x0'], res['x1'], res['z'], 100)
    plt.plot(x_exp[:, 0], x_exp[:, 1], '.', color='r', markersize=1)
    plt.plot(x[:, 0], x[:, 1], '-o', color='b', markersize=1)

    problem_f_type = problem_index // 15 + 1
    problem_index_type = problem_index_type_list[problem_index % 15]

    plt.title('2D_index_{} -- f_{:2d} id_{:2d}'.format(problem_index, problem_f_type, problem_index_type))

    fig.colorbar(cs)
    if save_fig:
        path_dir_fig = os.path.join(path_dir, 'figures')
        if not os.path.exists(path_dir_fig):
            os.makedirs(path_dir_fig)

        path_fig = os.path.join(path_dir_fig, '2D_index_{:3d}.pdf'.format(problem_index))
        plt.savefig(path_fig)
    else:
        plt.show()


    plt.close()

def plot_2D_contour_tmp(problem_index, save_fig=False):

    path_dir = os.path.join(base_dir, 'f_eval', '2D_Contour')
    path_res = os.path.join(path_dir, '2D_index_{}.npy'.format(problem_index))
    res = np.load(path_res).item()

    fig, ax = plt.subplots()
    cs = ax.contour(res['x0'], res['x1'], res['z'], 100)
    fig.colorbar(cs)
    plt.title("dim = 2 index = {}".format(problem_index))
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

    #merge_baseline_one_line_compare()
    #merge_bbo(optimizers=[], dimension=[2, 3, 5, 10, 20, 40, 784], plot_sum=False)
    #merge_bbo(optimizers=['value', 'first_order', 'second_order'], dimension=[1, 2, 3, 5, 10], plot_sum=False)
    dim = 2
    index = 0
    dir_name = 'CMP'


    path = os.path.join(base_dir, 'analysis', dir_name, str(dim))

    title = "{} dim = {} index = {}".format(dir_name, dim, index)
    #compare_pi_evaluate(dim, index, path, title, baseline_cmp=False)


    #plot_res(optimizers=["value", "first_order", "second_order", "anchor"], max_budget=12000, compare_baseline=True)

    for i in range(360):
        plot_1D(i, True)
        plot_2D_contour(i, True)
        plot_2D(i, save_fig=True)

    prefix = 'value_direct_3'
    path = os.path.join(base_dir, 'analysis', dir_name, str(dim), prefix)

    #plot_2D_contour(index, path, False)
    #plot_2D(15, path, False)
