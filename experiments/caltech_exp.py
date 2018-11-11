"""
Module for performing experiments over Caltech-101
"""
from abc import abstractmethod

import tensorflow as tf

from experiments.experiment import Experiment
from networks.alex_net import AlexNet
from input.data.caltech_data import CaltechData
import utils.constants as const


class CaltechExperiment(Experiment):
    """
    Performs experiments over Caltech-101 according to the User input and pre-established configurations
    """

    @abstractmethod
    def _prepare_trainer(self):
        pass

    def _prepare_data_pipeline(self):
        self.data_pipeline = CaltechData(self.general_config, self.train_dirs, self.validation_dir)

    def _prepare_neural_network(self):
        self.__input_tensor = tf.placeholder(tf.float32, [None, 227, 227, 3])
        self.__output_tensor = tf.placeholder(tf.float32, [None, 101])
        self.__neural_net = AlexNet({'data': self.input_tensor})

    @abstractmethod
    def _prepare_config(self, str_optimizer: str, is_incremental: bool):
        pass

    @property
    def dataset_name(self):
        return const.DATA_CALTECH_101

    @property
    def data_input(self):
        return self.data_pipeline

    @property
    def neural_net(self):
        return self.__neural_net

    @property
    def input_tensor(self):
        return self.__input_tensor

    @property
    def output_tensor(self):
        return self.__output_tensor
