"""
Module for performing tests over Imagenet
"""
import tensorflow as tf

from tests.tester import Tester
from models import CaffeNet
from training.train_conf import GeneralConfig, TrainConfig
from input.imagenet_data import ImagenetData


# TODO carga de checkpoints
class ImagenetTester(Tester):
    """
    Performs tests over Imagenet according to the User input and pre-established configurations
    """

    def _prepare_data_pipeline(self):
        self.data_pipeline = ImagenetData(self.general_config, self.train_dirs, self.validation_dir, self.extras)

    def _prepare_neural_network(self):
        self.__input_tensor = tf.placeholder(tf.float32, [None, 256, 256, 3])
        self.__output_tensor = tf.placeholder(tf.float32, [None, 200])
        self.__neural_net = CaffeNet({'data': self.input_tensor})

    def _prepare_config(self, str_optimizer: str):
        self.__general_config = GeneralConfig(self.lr, self.summary_interval, self.check_interval,
                                              config_name=str_optimizer, model_name='Imagenet')
        # Creates configuration for 5 mega-batches
        for i in range(5):
            train_conf = TrainConfig(1, batch_size=128)
            self.general_config.add_train_conf(train_conf)

    def _prepare_checkpoint_if_required(self, ckp_path: str):
        pass

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
    def checkpoint_loaded(self):
        return True

    @property
    def input_tensor(self):
        return self.__input_tensor

    @property
    def output_tensor(self):
        return self.__output_tensor
