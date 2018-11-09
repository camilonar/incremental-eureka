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
                     train_step: tf.Operation, loss: tf.Tensor, increment: int, iteration: int, total_it: int):
        return self.sess.run([self.train_step, self.loss],
                             feed_dict={self.tensor_x: image_batch, self.tensor_y: target_batch})

    def _create_loss(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        return tf.losses.softmax_cross_entropy(tensor_y, net_output)

    def _create_optimizer(self, config: GeneralConfig, loss: tf.Tensor, var_list=None):
        return tf.train.RMSPropOptimizer(config.learn_rate).minimize(loss, var_list=var_list)

