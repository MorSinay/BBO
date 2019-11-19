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

username = pwd.getpwuid(os.geteuid()).pw_name

if username == 'morsi':
    base_dir = os.path.join('/Users', username, 'Desktop', 'baseline')
else:
    from vae import VaeProblem, VAE
    base_dir = os.path.join('/data/', username, 'gan_rl', 'baseline')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

epsilon = 1

compare_grad = 1
compare_value = 1

dim = 40
index = 0
in_dir = 'Compare'

def run_baseline(budget=10000):
    suite_name = "bbob"
    suite_filter_options = ("dimensions: 2,3,5,10,20,40 ")  # "year:2019 " +  "instance_indices: 1-5 ")
    suite = cocoex.Suite(suite_name, "", suite_filter_options)

    optimization_function = [scipy.optimize.fmin_slsqp, scipy.optimize.fmin, scipy.optimize.fmin_cobyla, cma.fmin2]

    for fmin in optimization_function:
        data = defaultdict(list)
        suite = cocoex.Suite(suite_name, "", suite_filter_options)
        fmin_name = fmin.__name__

        for i, problem in tqdm(enumerate(suite)):

            if fmin_name is 'fmin_slsqp':
                output = fmin(problem, problem.initial_solution, iter=budget,  # very approximate way to respect budget
                              full_output=True, iprint=-1)

            elif fmin_name is 'fmin':
                output = fmin(problem, problem.initial_solution,
                              maxfun=budget * problem.dimension, disp=False, full_output=True)

            elif fmin_name is 'fmin2':
                xopt, es = fmin(problem, problem.initial_solution, 2,
                                {'maxfevals':budget * problem.dimension, 'verbose':-9}, restarts=9)

            elif fmin_name is 'fmin_cobyla':
                fmin(problem, problem.initial_solution, cons=lambda x: problem.constraint(x), maxfun=budget * problem.dimension,
                     disp=0, rhoend=1e-9)
            else:
                raise NotImplementedError

            data['index'].append(problem.index)
            data['hit'].append(problem.final_target_hit)
            data['id'].append(problem.id)
            data['dimension'].append(problem.dimension)
            data['best_observed'].append(problem.best_observed_fvalue1)
            data['initial_solution'].append(problem.initial_solution)
            data['upper_bound'].append(problem.upper_bounds)
            data['lower_bound'].append(problem.lower_bounds)
            data['number_of_evaluations'].append(problem.evaluations)

        df = pd.DataFrame(data)
        fmin_file = os.path.join(base_dir, 'coco_'+fmin_name+'.csv')
        df.to_csv(fmin_file)

def run_vae_baseline(budget=10000):

    optimization_function = [scipy.optimize.fmin_slsqp, scipy.optimize.fmin, scipy.optimize.fmin_cobyla, cma.fmin2]

    for fmin in optimization_function:
        data = defaultdict(list)
        fmin_name = fmin.__name__

        for i in tqdm(range(360)):
            vae_problem = VaeProblem(i)
            if fmin_name is 'fmin_slsqp':
                output = fmin(vae_problem.func, vae_problem.initial_solution, iter=budget,  # very approximate way to respect budget
                              full_output=True, iprint=-1)

            elif fmin_name is 'fmin':
                output = fmin(vae_problem.func, vae_problem.initial_solution,
                              maxfun=budget, disp=False, full_output=True)

            elif fmin_name is 'fmin2':
                xopt, es = fmin(vae_problem.func, vae_problem.initial_solution, 2,
                                {'maxfevals':budget, 'verbose':-9}, restarts=9)

            elif fmin_name is 'fmin_cobyla':
                fmin(vae_problem.func, vae_problem.initial_solution, cons=lambda x: vae_problem.constraint(x), maxfun=budget,
                     disp=0, rhoend=1e-9)


            else:
                raise NotImplementedError

            data['index'].append(i)
            data['hit'].append(vae_problem.final_target_hit)
            data['id'].append('vae' + str(i))
            data['dimension'].append(vae_problem.dimension)
            data['best_observed'].append(vae_problem.best_observed_fvalue1)
            data['initial_solution'].append(vae_problem.initial_solution)
            data['upper_bound'].append(vae_problem.upper_bounds)
            data['lower_bound'].append(vae_problem.lower_bounds)
            data['number_of_evaluations'].append(vae_problem.evaluations)

        df = pd.DataFrame(data)
        fmin_file = os.path.join(base_dir, 'vae_'+fmin_name+'.csv')
        df.to_csv(fmin_file)


def merge_baseline(optimizers=['fmin', 'fmin_slsqp', 'fmin2', 'fmin_cobyla']):

    data_fmin = pd.read_csv(os.path.join(base_dir, optimizers[0]+".csv"))
    data = data_fmin[['index', 'id', 'dimension', 'initial_solution', 'upper_bound', 'lower_bound']]

    hit_col = []
    for op in optimizers:
        data_fmin = pd.read_csv(os.path.join(base_dir, op+".csv"))
        data_fmin = data_fmin.rename(columns={"hit": op+"_hit", "best_observed": op+"_best_observed",
                                               "number_of_evaluations": op+"_number_of_evaluations"})
        hit_col.append(op+"_hit")

        data_fmin = data_fmin[['id', op+'_hit', op+'_best_observed', op+'_number_of_evaluations']]
        data = data.merge(data_fmin, on='id')

    data['baseline_hit'] = data[hit_col].max(axis=1)
    fmin_file = os.path.join(base_dir, 'baselines.csv')
    data.to_csv(fmin_file)

def merge_coco_vae(optimizers=['fmin', 'fmin_slsqp', 'fmin2', 'fmin_cobyla']):

    for i, op in enumerate(optimizers):
        coco_df = pd.read_csv(os.path.join(base_dir, 'coco_'+op+".csv"))
        vae_df = pd.read_csv(os.path.join(base_dir, 'vae_' + op + ".csv"))
        op_df = coco_df.append(vae_df)
        file = os.path.join(base_dir, op+'.csv')
        op_df.to_csv(file)


def merge_bbo(dim=['2', '3', '5', '10', '20', '40', '784'], optimizers=['fmin', 'fmin_slsqp', 'fmin2', 'fmin_cobyla']):

    if compare_grad and compare_value:
        bbo_options = ["bbo", "grad"]
    elif compare_value:
        bbo_options = ["bbo"]
    elif compare_grad:
        bbo_options = ["grad"]
    else:
        return

    df = [[] for _ in bbo_options]
    baseline_df = pd.read_csv(os.path.join(base_dir, 'baselines.csv'))
    best_observed = [op + '_best_observed' for op in optimizers]

    for i, op in enumerate(bbo_options):
        df[i] = pd.read_csv(os.path.join(base_dir, op+'_'+dim[0]+".csv"))

        for d in dim[1:]:
            df_d = pd.read_csv(os.path.join(base_dir, op+'_'+d+".csv"))
            df[i] = df[i].append(df_d, ignore_index=True)

        df[i] = df[i].rename(columns={"id":op+"_id", "hit": op+'_hit', "best_observed": op + '_best_observed', "number_of_evaluations": op + '_number_of_evaluations'})
        df[i] = df[i][[op+'_id', op+'_hit', op + '_best_observed', op + '_number_of_evaluations']]

        baseline_df = baseline_df.merge(df[i], left_on='id', right_on=op+'_id')#on='id')
        best_observed.append(op + '_best_observed')

        baseline_df[op + '_dist_from_min'] = np.abs(baseline_df[best_observed].min(axis=1) - baseline_df[op + '_best_observed'])
        baseline_df[op + '_dist_hit'] = baseline_df[op + '_dist_from_min'] < epsilon

    file = os.path.join(base_dir, 'compare.csv')
    baseline_df.to_csv(file)

# def problem_hit_map(optimizers=['fmin', 'fmin_slsqp', 'fmin2', 'fmin_cobyla', 'bbo', 'grad']):
#
#     baseline_df = pd.read_csv(os.path.join(base_dir, 'compare.csv'))
#     best_observed = [op + '_best_observed' for op in optimizers]
#
#     for i, op in enumerate(bbo_options):
#         df[i] = pd.read_csv(os.path.join(base_dir, op+'_'+dim[0]+".csv"))
#
#         for d in dim[1:]:
#             df_d = pd.read_csv(os.path.join(base_dir, op+'_'+d+".csv"))
#             df[i] = df[i].append(df_d, ignore_index=True)
#
#         df[i] = df[i].rename(columns={"id":op+"_id", "hit": op+'_hit', "best_observed": op + '_best_observed', "number_of_evaluations": op + '_number_of_evaluations'})
#         df[i] = df[i][[op+'_id', op+'_hit', op + '_best_observed', op + '_number_of_evaluations']]
#
#         baseline_df = baseline_df.merge(df[i], left_on='id', right_on=op+'_id')#on='id')
#         best_observed.append(op + '_best_observed')
#
#         baseline_df[op + '_dist_from_min'] = np.abs(baseline_df[best_observed].min(axis=1) - baseline_df[op + '_best_observed'])
#         baseline_df[op + '_dist_hit'] = baseline_df[op + '_dist_from_min'] < epsilon
#
#     file = os.path.join(base_dir, 'compare.csv')
#     baseline_df.to_csv(file)

def plot_res(optimizers=['fmin', 'fmin_slsqp', 'fmin2', 'fmin_cobyla', 'bbo', 'bbo__dist']):
    dimension = [2, 3, 5, 10, 20, 40, 784]
    color = ['b', 'g', 'r', 'y', 'c', 'm', 'k', '0.75']
    df = pd.read_csv(os.path.join(base_dir, "compare.csv"))

    res = [[] for _ in range (len(optimizers))]
    for _, dim in enumerate(dimension):
        df_temp = df[df['dimension'] == dim]
        #dim_size = 1.0*len(df_temp)
        for n, op in enumerate(optimizers):
            res[n].append(len(df_temp[df_temp[op+'_hit'] == 1]))

    X = [10*i for i in range(len(dimension))]
    ax = plt.subplot(111)
    w = 1

    for i in range(len(optimizers)):
        ax.bar([x + i*w for x in X], res[i], width=w, color=color[i], align='center', label=optimizers[i])

    ax.set_xticks([i + len(dimension)//2 for i in X])
    ax.set_xticklabels(dimension)
   # ax.autoscale(tight=True)
    ax.legend()

    plt.show()


def compare_beta_evaluate():
    compare_file = 'beta_evaluate.npy'#'best_observed.npy'
    analysis_path = os.path.join(base_dir, 'analysis', in_dir, str(dim))
    dirs = os.listdir(analysis_path)
    compare_dict = {}
    for dir in dirs:
        try:
            path = os.path.join(analysis_path, dir, str(index))
            files = os.listdir(path)
            if compare_file in files:
                compare_dict[dir] = np.load(os.path.join(path, compare_file))
        except:
            continue

    color = ['b', 'g', 'r', 'y', 'c', 'm', 'k', 'lime', 'gold', 'slategray', 'indigo', 'maroon', 'plum', 'pink', 'tan', 'khaki', 'silver',
             'navy', 'skyblue', 'teal', 'darkkhaki', 'indianred', 'orchid', 'lightgrey', 'dimgrey']
    res_keys = compare_dict.keys()
    if len(res_keys) is 0:
        return

    plt.subplot(111)

    for i, key in enumerate(res_keys):
        v = compare_dict[key]-79.48
        t = np.arange(1, compare_dict[key].size+1)
        #plt.plot(t, v.cumsum()/t, color=color[i], label=key)
        plt.plot(range(200), compare_dict[key][:200]-79.48, color=color[i], label=key)

    plt.title('dir: {} dim: {} index: {} compare: {}'.format(in_dir, dim, index, compare_file))
    plt.legend()
    plt.ylim([0, 200])
    plt.xlim([0, 200])
    plt.show()

def compare_baseline(optimizers):
    merge_coco_vae(optimizers=optimizers)
    merge_baseline(optimizers=optimizers)
    merge_bbo(dim=['2', '3', '5', '10', '20', '40', '784'], optimizers=optimizers)

    if compare_grad and compare_value:
        plot_res(optimizers=optimizers + ['bbo_dist', 'grad_dist'])
    elif compare_value:
        plot_res(optimizers=optimizers + ['bbo_dist'])
    elif compare_grad:
        plot_res(optimizers=optimizers + ['grad_dist'])


if __name__ == '__main__':
    #optimizers = ['fmin', 'fmin_slsqp', 'fmin_cobyla']
    optimizers = ['fmin', 'fmin_slsqp']
    #compare_baseline(optimizers)

    compare_beta_evaluate()

    #run_vae_baseline()