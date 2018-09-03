"""
Trainer with RMSProp Optimizer
"""
import tensorflow as tf

from training.train_conf import GeneralConfig
from training.trainer import Trainer


class RMSPropTrainer(Trainer):
    """
    Trainer with RMSProp Optimizer
    """

    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, mse: tf.Tensor, increment: int, iteration: int, total_it: int):
        return self.sess.run([self.train_step, self.mse],
                             feed_dict={self.tensor_x: image_batch, self.tensor_y: target_batch})

    def _create_mse(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        return tf.reduce_mean(tf.square(tensor_y - net_output))

    def _create_optimizer(self, config: GeneralConfig, mse: tf.Tensor):
        return tf.train.RMSPropOptimizer(config.learn_rate).minimize(mse)
