"""
The proposed algorithm that uses RMSProp and representatives selection for incremental learning (CRIL)
"""
from abc import ABC, abstractmethod

import math
import tensorflow as tf
import numpy as np

from etl.data import Data
from libs.caffe_tensorflow.network import Network
from training.config.cril_config import CRILConfig
from experiments.tester import Tester
from training.trainer.trainer import Trainer


class CRILTrainer(Trainer, ABC):
    """
    Trains with the proposed algorithm that uses RMSProp and representatives selection for incremental learning
    """

    def __init__(self, config: CRILConfig, model: Network, pipeline: Data, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                 tester: Tester = None, checkpoint: str = None):
        super().__init__(config, model, pipeline, tensor_x, tensor_y, tester=tester, checkpoint=checkpoint)

        self.representatives = [[] for _ in range(model.get_output().shape[1])]
        self.class_count = [0 for _ in range(model.get_output().shape[1])]
        self.weights = tf.placeholder(tf.float32, [None])

        self.buffered_reps = []
        self.memory_size = config.memory_size
        self.n_candidates = config.n_candidates
        self.buffer_size = config.buffer_size

        # TODO temporal placeholders for tests
        test = 1
        # Test for random with minimum values (1%)
        if test == 1:
            print("Test 1%")
            if self.config.model_name == 'FASHION-MNIST':
                self.memory_size = 50  # Maximum number of representatives per class
            elif self.config.model_name == 'CALTECH-101':
                self.memory_size = 1  # Maximum number of representatives per class
            else:
                self.memory_size = 50  # Maximum number of representatives per class

        # Tests for random maximum values (10%)
        elif test == 2:
            print("Test 10%")
            if self.config.model_name == 'FASHION-MNIST':
                self.memory_size = 500  # Maximum number of representatives per class
            elif self.config.model_name == 'CALTECH-101':
                self.memory_size = 10  # Maximum number of representatives per class
            else:
                self.memory_size = 500  # Maximum number of representatives per class

    def _create_loss(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        return tf.losses.softmax_cross_entropy(tf.multiply(tensor_y, self.mask_tensor),
                                               tf.multiply(net_output, self.mask_tensor), weights=self.weights)

    def _create_optimizer(self, config: CRILConfig, loss: tf.Tensor, var_list=None):
        return tf.train.RMSPropOptimizer(config.learn_rate).minimize(loss, var_list=var_list)

    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, loss: tf.Tensor, megabatch: int, iteration: int, total_it: int):
        # Gets the representatives
        reps = self.__get_representatives()
        n_reps = len(reps)

        # Gets the respective weights
        weights_values = np.full((len(image_batch)), 1.0)

        if n_reps > 0:
            rep_weights = [rep.weight for rep in reps]
            rep_values = [rep.value for rep in reps]
            rep_labels = [rep.label for rep in reps]
            # Concatenates the training samples with the representatives
            weights_values = np.concatenate((weights_values, rep_weights))
            image_batch = np.concatenate((image_batch, rep_values))
            target_batch = np.concatenate((target_batch, rep_labels))

        # Executes the update of the net
        ts, loss, outputs = self.sess.run([self.train_step, self.loss, self.model.get_output()],
                                          feed_dict={self.tensor_x: image_batch, self.tensor_y: target_batch,
                                                     self.weights: weights_values,
                                                     self.mask_tensor: self.mask_value})

        # Modifies the list of representatives (random)
        if n_reps == 0:
            self._buffer_candidates(image_batch, target_batch, outputs, total_it, megabatch)
        else:
            self._buffer_candidates(image_batch[:-n_reps], target_batch[:-n_reps], outputs[:-n_reps], total_it,
                                    megabatch)
        if total_it % self.buffer_size == 0:
            self._modify_representatives(self.buffered_reps)
            self.__recalculate_weights(self.representatives)
            self.__clear_buffer()

        return ts, loss

    def __get_representatives(self):
        """
        Selects or retrieves the representatives from the data

        :return: a list of representatives.
            The method returns an empty array **[]** if the number of representatives is less than the minimum
            number of representatives per batch (rep_per_batch)
        """
        repr_list = np.concatenate(self.representatives)
        if repr_list.size > 0:
            samples = np.random.choice(repr_list, size=min(self.n_candidates, repr_list.size), replace=False)
            return samples
        else:
            return []

    @abstractmethod
    def _buffer_candidates(self, image_batch, target_batch, outputs, iteration, megabatch):
        """
        Adds samples to the buffer of candidates.

        :param image_batch: the list of images of a batch
        :param target_batch: the list of one hot labels of a batch
        :param outputs: output probabilities of the neural network
        :param iteration: current iteration of training
        :param megabatch: current megabatch

        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the _buffer_samples")

    @abstractmethod
    def _modify_representatives(self, candidate_representatives):
        """
        Modifies the representatives list according to the new data (i.e. the candidate representatives)
        
        :param candidate_representatives: the preselected representatives from the buffer
        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the _modify_representatives")

    def __clear_buffer(self):
        """
        Clears the buffer
        :return: None
        """
        self.buffered_reps = []

    def __recalculate_weights(self, representatives):
        """
        Reassigns the weights of the representatives
        :param representatives: a list of representatives
        :return: None
        """
        total_count = np.sum(self.class_count)
        # This version proposes that the total weight of representatives is calculated from the proportion of candidate
        # representatives respect to the batch. E.g. a batch of 100 images and 10 are preselected, total_weight = 10
        total_weight = (self.config.train_configurations[0].batch_size * 1.0) / self.n_candidates
        # The total_weight is adjusted to the proportion between candidate representatives and actual representatives
        total_weight *= (total_count / np.sum([len(cls) for cls in representatives]))
        probs = [count / total_count for count in self.class_count]
        for i in range(len(representatives)):
            for rep in representatives[i]:
                # This version uses log as an stabilizer
                rep.weight = max(math.log(probs[i] * total_weight), 1.0)


class Representative(object):
    """
    Representative sample of the algorithm
    """

    def __init__(self, value, label, metric, iteration, megabatch, net_output=None, crowd_distance=None):
        """
        Creates a Representative object
        :param value: the value of the representative (i.e. the image)
        :param label: the expected ground truth label (in one-hot format)
        :param metric: the value of the metric
        :param iteration: the iteration at which the sample was selected as representative
        :param megabatch: the current megabatch
        :param net_output: the output that the neural network gives to the sample
        :param crowd_distance: a measure of distance to the other representatives of the same cluster (e.g. same class)
        """
        self.value = value
        self.label = label
        self.metric = metric
        self.iteration = iteration
        self.net_output = net_output
        self.crowd_distance = crowd_distance
        self.megabatch = megabatch
        self.weight = 3.0

    def __eq__(self, other):
        if isinstance(other, Representative.__class__):
            return self.value.__eq__(other.value)
        return False
