"""
The proposed algorithm that uses RMSProp and representatives selection for incremental learning
"""
import tensorflow as tf
import numpy as np

from input.data import Data
from libs.caffe_tensorflow.network import Network
from training.general_config import GeneralConfig
from training.trainer import Trainer


# TODO implementar
class RepresentativesTrainer(Trainer):
    """
    Trains with the proposed algorithm that uses RMSProp and representatives selection for incremental learning
    """

    def __init__(self, config: GeneralConfig, model: Network, pipeline: Data, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                 checkpoint: str = None):
        super().__init__(config, model, pipeline, tensor_x, tensor_y, checkpoint)
        self.representatives = tf.placeholder(tf.float32, tensor_x.get_shape())
        self.weights = tf.placeholder(tf.float32, [None])

    def _create_loss(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        return tf.losses.softmax_cross_entropy(tensor_y, net_output, weights=self.weights)

    def _create_optimizer(self, config: GeneralConfig, loss: tf.Tensor, var_list=None):
        return tf.train.RMSPropOptimizer(config.learn_rate).minimize(loss, var_list=var_list)

    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, loss: tf.Tensor, increment: int, iteration: int, total_it: int):
        # Gets the representatives
        rep_values, rep_labels = self.__get_representatives(sess, image_batch, target_batch,
                                                            tensor_x, tensor_y)

        # Gets the respectives weights
        weights_values = np.full((len(image_batch)), 1.0)
        rep_weights = np.full((len(rep_values)), 3.0)
        weights_values = np.concatenate((weights_values, rep_weights))

        # Concatenates the training samples with the representatives
        image_batch = np.concatenate((image_batch, rep_values))
        target_batch = np.concatenate((target_batch, rep_labels))
        # Executes the update of the net
        return self.sess.run([self.train_step, self.loss],
                             feed_dict={self.tensor_x: image_batch, self.tensor_y: target_batch,
                                        self.weights: weights_values})

    # TODO implementar. Por ahora retorna representantes aleatorios. Evaluar: uso de fronteras y centros
    def __get_representatives(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor):
        """
        Selects or retrieves the representatives from the data
        :param sess: the current session
        :param image_batch: he batch of data corresponding to the input, as obtained from the data pipeline
        :param target_batch: the batch of data corresponding to the output, as obtained from the data pipeline
        :param tensor_x: the tensor corresponding to the input of a training
        :param tensor_y: the tensor corresponding to the output of a training
        :return: a tuple with 2 numpy.ndarray with the data and the labels. The data array has shape
        [n_representatives, x1, x2, ..., xn] where [x1...xn] is the shape of a single sample image. The labels
        array has shape [n_representatives, n_labels].
        """
        # outputs = sess.run(model, feed_dict={self.tensor_x: image_batch, self.tensor_y: target_batch})
        # dif = outputs[0][outputs.shape[1] - 1] - outputs[0][outputs.shape[1] - 2]
        random = np.random.choice(image_batch.shape[0], 20, replace=False)
        return image_batch[random, :], target_batch[random, :]
