"""
Module for performing tests over Imagenet
"""
from abc import abstractmethod

import tensorflow as tf

from tests.tester import Tester
from models import CaffeNet
from input.data.imagenet_data import ImagenetData
import utils.constants as const


class ImagenetTester(Tester):
    """
    Performs tests over Imagenet according to the User input and pre-established configurations
    """

    @abstractmethod
    def _prepare_trainer(self):
        pass

    def _prepare_data_pipeline(self):
        self.data_pipeline = ImagenetData(self.general_config, self.train_dirs, self.validation_dir, self.extras)

    def _prepare_neural_network(self):
        self.__input_tensor = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.__output_tensor = tf.placeholder(tf.float32, [None, 200])
        self.__neural_net = CaffeNet({'data': self.input_tensor})

    @abstractmethod
    def _prepare_config(self, str_optimizer: str, is_incremental: bool):
        pass

    @property
    def dataset_name(self):
        return const.DATA_TINY_IMAGENET

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
