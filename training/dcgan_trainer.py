"""
Module for the training algorithm that uses artificial sampling with DCGAN
"""
import tensorflow as tf

from training.train_conf import GeneralConfig
from training.trainer import Trainer


# TODO implementar
class DCGANTrainer(Trainer):
    """
    Trainer that uses the algorithm presented in "Evolutive deep models for online learning on data
    streams with no storage"
    See: http://ceur-ws.org/Vol-1958/IOTSTREAMING2.pdf
    """

    def _create_loss(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        pass

    def _create_optimizer(self, config: GeneralConfig, loss: tf.Tensor):
        pass

    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, loss: tf.Tensor, increment: int, iteration: int, total_it: int):
        pass