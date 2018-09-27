"""
Module for performing tests over CIFAR-10
"""
from abc import abstractmethod

import tensorflow as tf

from tests.tester import Tester
from models import CifarTFNet
from input.cifar_data import CifarData
import utils.constants as const


class CifarTester(Tester):
    """
    Performs tests over CIFAR-10 according to the User input and pre-established configurations
    """
    @abstractmethod
    def _prepare_trainer(self):
        pass

    def _prepare_data_pipeline(self):
        self.data_pipeline = CifarData(self.general_config, self.train_dirs, self.validation_dir, self.extras)

    def _prepare_neural_network(self):
        self.__input_tensor = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.__output_tensor = tf.placeholder(tf.float32, [None, 10])
        self.__neural_net = CifarTFNet({'data': self.input_tensor})

    @abstractmethod
    def _prepare_config(self, str_optimizer: str):
        pass

    @property
    def dataset_name(self):
        return const.DATA_CIFAR_10

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
