"""
Module for performing tests over CIFAR-10
"""
from abc import abstractmethod

import tensorflow as tf

from tests.tester import Tester
from models import NiN
from models import CifarTFNet
from input.cifar100_data import Cifar100Data
import utils.constants as const


class Cifar100Tester(Tester):
    """
    Performs tests over CIFAR-100 according to the User input and pre-established configurations
    """
    @abstractmethod
    def _prepare_trainer(self):
        pass

    def _prepare_data_pipeline(self):
        self.data_pipeline = Cifar100Data(self.general_config, self.train_dirs, self.validation_dir, self.extras)

    def _prepare_neural_network(self):
        self.__input_tensor = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.__output_tensor = tf.placeholder(tf.float32, [None, 100])
        self.__neural_net = NiN({'data': self.input_tensor})

    @abstractmethod
    def _prepare_config(self, str_optimizer: str):
        pass

    @property
    def dataset_name(self):
        return const.DATA_CIFAR_100

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
