"""
Module with utility functions for directory management
"""
from datetime import datetime
import os
import utils as utils
from training.train_conf import GeneralConfig


def get_unique_logdir():
    """
    Creates an unique logging directory based on current datetime (year, month, ..., seconds)
    :return: a String with a directory name. The returned value DOES NOT have any OS specific separator, e.g.
    it would return 'log_2018-10-04-49-24' and not 'log_2018-10-04-49-24/'
    """
    timestamp = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
    return 'log_{}'.format(timestamp)


"""
Module used for training of a convolutional NN with TensorFlow. It has some utilities for using
checkpoints, logging and summaries.
"""


def prepare_directories(config: GeneralConfig):
    """
    Creates and prepares the directories of checkpoints and summaries for TensorBoard
    :param config: the configuration that is going to be used in the whole training
    :return: a tuple containing a path for a checkpoint directory and a path for a summaries directory
    """
    ckpt_path = os.path.join('checkpoints', config.model_name, config.config_name)
    summaries_path = os.path.join('summaries', config.model_name, config.config_name, get_unique_logdir())

    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

    if not os.path.isdir(summaries_path):
        os.makedirs(summaries_path)

    return ckpt_path, summaries_path
