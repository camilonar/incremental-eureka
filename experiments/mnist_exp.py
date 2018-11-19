"""
Module for performing experiments over MNIST
"""
from abc import abstractmethod

import tensorflow as tf

from experiments.experiment import Experiment
from networks.le_net import LeNet
from input.data.mnist_data import MnistData
import utils.constants as const


class MnistExperiment(Experiment):
    """
    Performs experiments over MNIST according to the User input and pre-established configurations
    """

    @abstractmethod
    def _prepare_trainer(self):
        pass

    def _prepare_data_pipeline(self):
        self.data_pipeline = MnistData(self.general_config, self.train_dirs, self.validation_dir)

    def _prepare_neural_network(self):
        self.__input_tensor = tf.placeholder(tf.float32, [None, 32, 32, 1])
        self.__output_tensor = tf.placeholder(tf.float32, [None, 10])
        self.__neural_net = LeNet({'data': self.input_tensor})

    @abstractmethod
    def _prepare_config(self, str_optimizer: str, is_incremental: bool):
        pass

    @property
    def dataset_name(self):
        return const.DATA_MNIST

    @property
    def data_input(self):
        return self.data_pipeline

    @property
    def neural_net(self):
        return self.__neural_net

    @property
    @abstractmethod
    def general_config(self):
        pass

    @property
    def input_tensor(self):
        return self.__input_tensor

    @property
    def output_tensor(self):
        return self.__output_tensor