"""
Module for the CEAL training algorithm
"""
import numpy
import tensorflow as tf
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

from input.data import Data
from network import Network
from training.ceal_conf import CealConfig
from training.trainer import Trainer


class CEALTrainer(Trainer):
    """
    Trainer that uses the CEAL algorithm
    See: Keze Wang, Dongyu Zhang, Ya Li, Ruimao Zhang, and Liang Lin.
        Cost-effective active learning for deep image classification.
        IEEE Transactions on Circuits and Systems for Video Technology, 2016)
    """

    def __init__(self, config: CealConfig, model: Network, pipeline: Data, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
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

        self.du = None
        self.du_y = None
        self.dl_x = None
        self.dl_y = None
        self.dh_x = None
        self.dh_y = None

        # Unlabeled set
        # Labeled set

    def _create_mse(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        return tf.reduce_mean(tf.square(tensor_y - net_output))

    def _custom_prepare(self, sess):
        # TODO
        pass

    def _create_optimizer(self, config: CealConfig, mse: tf.Tensor):
        return GradientDescentOptimizer(config.learn_rate).minimize(mse)

    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, mse: tf.Tensor, increment: int, iteration: int, total_it: int):

        if increment == 0:
            self.dl_x = image_batch
            self.dl_y = target_batch
            return self.sess.run([self.train_step, self.mse],
                                 feed_dict={self.tensor_x: image_batch, self.tensor_y: target_batch})
        else:

            self.du = image_batch
            self.du_y = target_batch  # represent manual tag
            predicts = self.__get_prediction(self.sess,self.du,self.tensor_x)
            predicts_entropy = self.__calculate_entropy(predicts)
            # Add K uncertainty samples into DL
            # TODO: select method of sampling
            lc_x, lc_y = self.__get_least_confidence_sampling(self.config.k,predicts)
            self.dl_x = numpy.append(self.dl_x, lc_x)
            self.dl_y = numpy.append(self.dl_y, lc_y)
            # Obtain high conﬁdence samples DH
            self.dh_x, self.dh_y = self.__get_high_confidence_sampling(predicts,predicts_entropy)
            # In every t iterations
            if True:
                # self.config.delta = self.config.delta - self.config.dr * t
                return self.sess.run([self.train_step, self.mse],
                                     feed_dict={self.tensor_x: image_batch, self.tensor_y: target_batch})

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

    def __get_least_confidence_sampling(self, k: int, predicts: numpy.ndarray):
        """
        Gets the k least confident samples. A sample has low confidence if the probability of the most probable class
        is low
        :param k: the number of samples to be retrieved
        :return: a numpy array with the k least confident unlabeled samples
        """
        predicts_max = numpy.amax(predicts, axis=1)
        index_sorted = numpy.argpartition(predicts_max, k)
        return numpy.take(self.du, index_sorted[0:k]), numpy.take(self.du_y, index_sorted[0:k])

    def __get_margin_sampling(self, k: int, predicts: numpy.ndarray):
        """
        Gets the k least confident samples according to the Best vs. Second Best strategy, that is, according to the
        difference between the probability of the most probable class and the probability of the second most probable
        class to which the sample may belong
        :param k: the number of samples to be retrieved
        :return: a numpy array with the k least confident unlabeled samples
        """
        delt_preds = numpy.array([])
        for pred in predicts:
            temp = -numpy.partition(-pred, 2)
            delta = temp[0] - temp[1]
            delt_preds = numpy.append(delt_preds, delta)
        index_sorted = numpy.argpartition(delt_preds, k)
        return numpy.take(self.du, index_sorted[0:k]), numpy.take(self.du_y, index_sorted[0:k])

    def __get_entropy_sampling(self, k: int,entropy_predicts: numpy.ndarray):
        """
        Gets the k least confident samples according to the Entropy measure, which is calculated using all the class
        label probabilities. A sample with big entropy is an uncertain sample
        :param k: the number of samples to be retrieved
        :return: a numpy array with the k least confident unlabeled samples
        """
        index_sorted = numpy.argpartition(-entropy_predicts, k)  # sort asc
        return numpy.take(self.du, index_sorted[0:k]), numpy.take(self.du_y, index_sorted[0:k])

    def __get_high_confidence_sampling(self, predicts: numpy.ndarray, predicts_entropy: numpy.ndarray):
        '''
        get samples that have high confidence of unlabeled samples
        :return: a numpy array with high confidence samples
        '''
        indexs_high_confidence = numpy.where(predicts_entropy < self.config.delta)
        dh_x = numpy.take(self.du, indexs_high_confidence)
        dh_y = tf.one_hot(numpy.argmax(numpy.take(predicts, indexs_high_confidence)), predicts.shape[0])
        return dh_x, dh_y

    def __calculate_entropy(self, predicts: numpy.ndarray):
        """
        Calculates the entropy of a sample
        :param predicts: the samples to which the entropy is going to be calculated
        :return: numpy array with values of entropy
        """
        return -sum(predicts * numpy.log(predicts))  # calculate entropy

    # TODO implementar mecanismo para hacer un guardado y carga adecuada de todos los datos de un checkpoint
