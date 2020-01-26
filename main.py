from config import args, exp
import numpy as np
from loguru import logger
import pandas as pd
import os
from bbo import BBO
from tqdm import tqdm
from collections import defaultdict
from baseline import Baseline


def get_algorithm():

    if args.algorithm in ['egl', 'igl']:
        return BBO()
    raise NotImplementedError


def reload(alg):

    aux = defaultdict(lambda: 0)
    if exp.load_model and args.reload:
        try:
            aux = alg.load_checkpoint(exp.checkpoint)
        except Exception as e:
            logger.error(str(e))

    return aux


def baseline():

    b = Baseline()
    train_results = b.learn()

    exp.log_data(train_results, n=1, alg=None)


def optimize():

    alg = get_algorithm()

    aux = reload(alg)
    n_offset = aux['n']

    for epoch, train_results in enumerate(alg.optimize()):
        n = n_offset + (epoch + 1) * args.train_epoch

        exp.log_data(train_results, n=n, alg=alg if args.lognet else None)

        aux = {'n': n}
        alg.save_checkpoint(exp.checkpoint, aux)


def main():

    if args.optimize:
        logger.info("Optimization session")
        optimize()

    elif args.baseline:
        logger.info("Baseline session")
        baseline()

    else:
        raise NotImplementedError

    logger.info("End of simulation")
    exp.exit()


if __name__ == '__main__':
    main()

