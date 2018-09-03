"""
Module for the CEAL training algorithm
"""
import numpy
import tensorflow as tf
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

from input.data import Data
from network import Network
from training.train_conf import GeneralConfig
from training.trainer import Trainer


class CEALTrainer(Trainer):
    """
    Trainer that uses the CEAL algorithm
    See: Keze Wang, Dongyu Zhang, Ya Li, Ruimao Zhang, and Liang Lin.
        Cost-effective active learning for deep image classification.
        IEEE Transactions on Circuits and Systems for Video Technology, 2016)
    """
    def __init__(self, config: GeneralConfig, model: Network, pipeline: Data, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                 checkpoint: str = None):
        """
        It creates a CEALTrainer object
        :param config: the configuration for the whole training
        :param model: the neural net that is going to be trained
        :param pipeline: the data pipeline for the training
        :param checkpoint: the checkpoint path if it's required to start the training from a checkpoint. A data path with
        the following structure is expected: ./checkpoints/dataset_name/config_net_name/checkpoint_name.ckpt.
        If there is no checkpoint to be loaded then its value should be None. The default value is None.
        """
        super().__init__(config, model, pipeline, tensor_x, tensor_y, checkpoint)
        # TODO
        # uncertain samples selection size
        # High confidence selection threshold
        # Threshold decay rate
        # Fine tuning interval
        # Selection criteria

        # Unlabeled set
        # Labeled set

    def _create_mse(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        return tf.reduce_mean(tf.square(tensor_y - net_output))

    def _custom_prepare(self, sess):
        # TODO
        pass

    def _create_optimizer(self, config: GeneralConfig, mse: tf.Tensor):
        return GradientDescentOptimizer(config.learn_rate).minimize(mse)

    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, mse: tf.Tensor, increment: int, iteration: int, total_it: int):
        # TODO
        pass

    def __get_prediction(self, sess, samples: numpy.ndarray, tensor_x: tf.Tensor):
        """
        Gets the classification of a group of samples given by the output of the current model
        :param sess: the current session
        :param samples: the sample or samples that are going to be classified
        :param tensor_x: the tensor corresponding to the input of the neural network
        :return: a numpy array with the values for all the labels. An element of this array is of the same size as the
        output layer of the neural network, and there is one element for each sample
        """
        return sess.run(self.model.get_output(), feed_dict={tensor_x: samples})

    def __get_least_confidence_sampling(self, k: int):
        """
        Gets the k least confident samples. A sample has low confidence if the probability of the most probable class
        is low
        :param k: the number of samples to be retrieved
        :return: a numpy array with the k least confident unlabeled samples
        """
        pass

    def __get_margin_sampling(self, k: int):
        """
        Gets the k least confident samples according to the Best vs. Second Best strategy, that is, according to the
        difference between the probability of the most probable class and the probability of the second most probable
        class to which the sample may belong
        :param k: the number of samples to be retrieved
        :return: a numpy array with the k least confident unlabeled samples
        """
        pass

    def __get_entropy_sampling(self, k: int):
        """
        Gets the k least confident samples according to the Entropy measure, which is calculated using all the class
        label probabilities. A sample with big entropy is an uncertain sample
        :param k: the number of samples to be retrieved
        :return: a numpy array with the k least confident unlabeled samples
        """
        pass

    def __calculate_entropy(self, input_array: numpy.ndarray):
        """
        Calculates the entropy of a sample
        :param input_array: the sample to which the entropy is going to be calculated
        :return: a float with the entropy of a sample
        """
        pass

    # TODO implementar mecanismo para hacer un guardado y carga adecuada de todos los datos de un checkpoint
