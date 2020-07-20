"""
Trainer with RMSProp Optimizer
"""
import tensorflow as tf

from training.config.general_config import GeneralConfig
from training.trainer.trainer import Trainer


class RMSPropTrainer(Trainer):
    """
    Trainer with RMSProp Optimizer
    """

    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, loss: tf.Tensor, megabatch: int, iteration: int, total_it: int):
        return sess.run([train_step, loss],
                        feed_dict={tensor_x: image_batch, tensor_y: target_batch,
                                   self.mask_tensor: self.mask_value})

    def _precreate_loss(self):
        return tf.losses.softmax_cross_entropy, {}

    def _create_optimizer(self, config: GeneralConfig, loss: tf.Tensor, var_list=None):
        return tf.train.RMSPropOptimizer(config.learn_rate).minimize(loss, var_list=var_list)
