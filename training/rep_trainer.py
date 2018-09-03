"""
The proposed algorithm that uses RMSProp and representatives selection for incremental learning
"""
import tensorflow as tf

from training.train_conf import GeneralConfig
from training.trainer import Trainer


# TODO implementar
class RepresentativesTrainer(Trainer):
    """
    Trains with the proposed algorithm that uses RMSProp and representatives selection for incremental learning
    """

    def _create_mse(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        pass

    def _create_optimizer(self, config: GeneralConfig, mse: tf.Tensor):
        pass

    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, mse: tf.Tensor, increment: int, iteration: int, total_it: int):
        pass
