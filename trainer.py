"""
Module for training a neural network
"""
import tensorflow as tf
import numpy

from input.data import Data
from network import Network
from train_conf import GeneralConfig, TrainConfig
import utils


class Trainer(object):
    """
    Has the purpose of training a generic net, with a generic configuration and optimizer
    """

    def __init__(self, config: GeneralConfig, model: Network, pipeline: Data, tensor_x: tf.Tensor, tensor_y: tf.Tensor):
        """
        It creates a Trainer object
        :param config: the configuration for the whole training
        :param model: the neural net that is going to be trained
        :param pipeline: the data pipeline for the training
        """
        self.config = config
        self.model = model
        self.pipeline = pipeline
        self.tensor_x = tensor_x
        self.tensor_y = tensor_y

    def __prepare(self):
        """
        It does the preparation for the training
        :return: None
        """
        ck_path, summaries_path = utils.prepare_directories(self.config)
        # TODO crear sesión
        # TODO merge de summaries
        # TODO carga de checkpoints
        # TODO otras preparaciones

    # TODO hacerlo genérico para cualquier Optimizer
    # TODO ver manejo de datos
    # TODO si hay checkpoint, cómo se va a manejar el restablecimiento del batch correcto de datos? Es decir, se debe reestablecer el mega batch correcto
    def train(self):
        """
        Trains a neural network with the appropriate configuration.
        :return: None
        """
        self.__prepare()
        for i, _ in enumerate(self.config.train_configurations):
            data_x, data_y = self.pipeline.build_train_data_tensor()
            self.train_increment(self.config, data_x, data_y)
            if i < len(self.config.train_configurations):
                self.pipeline.change_dataset_part(i + 1)

    # TODO hacerlo genérico para cualquier Optimizer
    # TODO NOTA: no es necesario pasar el número de iteraciones o epochs, porque dataset saca OutOfRangeError automáticamente cuando termina de recorrer los datos
    def train_increment(self, config: GeneralConfig, data_x: tf.Tensor, data_y: tf.Tensor):
        """
        Trains a neural network over an increment of data, with the given configuration and model, over the data that is
        provided. It also saves summaries for TensorBoard and creates checkpoints according to the given configuration.
        :param config: the configuration for the whole training
        :param data_x: the tensor associated with the training data
        :param data_y: the tensor that has the corresponding labels of the data
        :return: None
        """
        pass
