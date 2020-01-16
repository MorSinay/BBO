from config import args, exp
import numpy as np
from loguru import logger
import pandas as pd
import os
from bbo import BBO
from tqdm import tqdm
from collections import defaultdict


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


def optimize(alg):
    aux = reload(alg)
    n_offset = aux['n']

    for epoch, train_results in enumerate(alg.optimize()):
        n = n_offset + (epoch + 1) * args.train_epoch

        exp.log_data(train_results, n=n, alg=alg if args.lognet else None)

        aux = {'n': n}
        alg.save_checkpoint(exp.checkpoint, aux)


def main():

    alg = get_algorithm()

    if args.optimize:
        logger.info("Optimization session")
        optimize(alg)

    else:
        raise NotImplementedError

    logger.info("End of simulation")


if __name__ == '__main__':
    main()

