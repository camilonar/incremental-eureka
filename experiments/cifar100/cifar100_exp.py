"""
Module for performing experiments over CIFAR-10
"""
from abc import abstractmethod, ABC

import tensorflow as tf

from experiments.experiment import Experiment
from networks.nin import NiN
from networks.simple_net import SimpleNet
from input.data.cifar100_data import Cifar100Data
import utils.constants as const


class Cifar100Experiment(Experiment, ABC):
    """
    Performs experiments over CIFAR-100 according to the User input and pre-established configurations
    """
    dataset_name = const.DATA_CIFAR_100
    data_input = None
    neural_net = None
    input_tensor = None
    output_tensor = None

    def _prepare_data_pipeline(self):
        self.data_input = Cifar100Data(self.general_config, self.train_dirs, self.validation_dir)

    def _prepare_neural_network(self):
        self.input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.output_tensor = tf.placeholder(tf.float32, [None, 100])
        self.neural_net = SimpleNet({'data': self.input_tensor})
