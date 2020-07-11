"""
Module for performing experiments over Imagenet
"""
from abc import abstractmethod, ABC

import tensorflow as tf

from experiments.experiment import Experiment
from etl.data.imagenet_data import ImagenetData
import utils.constants as const
from networks.alex_net import AlexNet


class ImagenetExperiment(Experiment, ABC):
    """
    Performs experiments over Imagenet according to the User input and pre-established configurations
    """
    dataset_name = const.DATA_TINY_IMAGENET
    data_input = None
    neural_net = None

    def _prepare_data_pipeline(self):
        self.data_input = ImagenetData(self.general_config, self.train_dirs, self.validation_dir)

    def _prepare_neural_network(self):
        self.neural_net = AlexNet({'data': self.input_tensor}, 200)
