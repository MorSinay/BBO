import logging
import time
import os
from config import args, consts
import sys


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):

    logger = None
    logging.shutdown()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    # File Handler
    filename = os.path.join(consts.logdir, "%d_%s.log" % (args.action_space, consts.exptime))
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    @staticmethod
    def info(*args, **kwargs):
        return Logger.logger.info(*args, **kwargs)

    @staticmethod
    def error(*args, **kwargs):
        return Logger.logger.error(*args, **kwargs)

    @staticmethod
    def debug(*args, **kwargs):
        return Logger.logger.debug(*args, **kwargs)

    @staticmethod
    def warning(*args, **kwargs):
        return Logger.logger.warning(*args, **kwargs)


logger = Logger()