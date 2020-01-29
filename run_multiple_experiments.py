import os
import subprocess
from collections import defaultdict
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

aux_args = ['--budget=10000', '--architecture=spline', '--explore=ball']

# images = [52530, 172895, 43849, 32973, 8527, 112953, 13363, 148362, 173380,
#           24810, 124309, 169689, 27201, 27430, 55602, 163705, 97915, 175273]

# full mini-dataset
images = [40001, 52530, 172895, 43849, 32973, 8527, 148362, 173380,
          24810, 124309, 169689, 27201, 27430, 55602, 163705, 97915,
          175273, 130369, 7536, 188494, 123363, 93787, 198999, 73333,
          112621, 77850, 145451, 177451, 196575, 165372, 171934, 191893,
          199013, 172359]

# ignore = [(40001, 'egl')]
ignore = []

# return to run_30_fc_org_att
# images = [112621, 77850, 145451, 177451, 196575, 165372, 171934, 191893,
#           199013, 172359]


# algorithms = [('--optimize', 'egl'), ('--optimize', 'igl'), ('--baseline', 'cma'), ('--baseline', 'cg')]
algorithms = [('--optimize', 'egl'), ('--optimize', 'igl')]
exp_name = 'run_30_spline_ball'
num = 0
# data_path = '/home/mlspeech/elads/data/landmarks/results'


def run_algorithms():

    for image in images:
        for method, alg in algorithms:

            if (image, alg) in ignore:
                continue

            print(f"image: {image} | algorithm: {alg}")
            arguments = [method,  f'--algorithm={alg}', f'--identifier={exp_name}',
                         f'--problem-index={image}',  f'--num={num}'] + aux_args

            subprocess.run(['python', 'main.py', *arguments])


# def visualization():
#
#     data = defaultdict(lambda: defaultdict(lambda: OrderedDict()))
#
#     for method, alg in algorithms:
#         for image in images:
#
#             # igl_debug_net_celeba_p_59802_exp_0001
#             root_pattern = f"{alg}_{exp_name}_celeba_p_{image}_exp_{num:04d}"
#
#             full_name = None
#             for name in os.listdir(data_path):
#
#                 # if name == 'egl_run10_celeba_p_52530_exp_0000_20200123_111247':
#                 #     print('xxx')
#
#                 if root_pattern in name:
#                     full_name = name
#                     break
#
#             if full_name is None:
#                 continue
#
#             exp_path = os.path.join(data_path, full_name, 'results', 'train')
#
#             for name in os.listdir(exp_path):
#                 n = int(name.split('.')[0])
#                 data[alg][image][n] = np.load(os.path.join(exp_path, exp_path, name), allow_pickle=True).item()
#
#
#     # collect best images:
#
#     best_images = {}
#
#     for method, alg in algorithms:
#
#         best_image = []
#
#         for image in solved_images:
#             n, d = list(data[alg][image].items())[-1]
#
#             best_image.append(d['image']['best_image'].permute(1, 2, 0))
#
#         best_images[alg] = np.hstack(best_image)
#
#     best_images = np.vstack([best_images[alg] for method, alg in algorithms])
#     plt.imshow(best_images)
#     # plt.savefig()
#     print("exit")


def main():

    run_algorithms()
    # visualization()

if __name__ == '__main__':
    main()
