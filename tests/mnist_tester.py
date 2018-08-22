"""
Module for performing tests over MNIST
"""
import tensorflow as tf

from tests.tester import Tester
from models import LeNet
from training.train_conf import GeneralConfig, TrainConfig
from input.mnist_data import MnistData
import utils.constants as const


class MnistTester(Tester):
    """
    Performs tests over MNIST according to the User input and pre-established configurations
    """

    def _prepare_data_pipeline(self):
        self.data_pipeline = MnistData(self.general_config, self.train_dirs, self.validation_dir, self.extras)

    def _prepare_neural_network(self):
        self.__input_tensor = tf.placeholder(tf.float32, [None, 28 , 28])
        self.__output_tensor = tf.placeholder(tf.float32, [None, 10])
        self.__neural_net = LeNet({'data': self.input_tensor})

    def _prepare_config(self, str_optimizer: str):
        self.__general_config = GeneralConfig(self.lr, self.summary_interval, self.ckp_interval,
                                              config_name=str_optimizer, model_name=self.dataset_name)
        # Creates configuration for 5 mega-batches
        for i in range(5):
            train_conf = TrainConfig(1, batch_size=128)
            self.general_config.add_train_conf(train_conf)

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
    def general_config(self):
        return self.__general_config

    @property
    def input_tensor(self):
        return self.__input_tensor

    @property
    def output_tensor(self):
        return self.__output_tensor