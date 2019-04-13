"""
The proposed algorithm that uses RMSProp and representatives selection for incremental learning
"""
import math
import tensorflow as tf
import numpy as np

from etl.data import Data
from libs.caffe_tensorflow.network import Network
from training.config.general_config import GeneralConfig
from experiments.tester import Tester
from training.trainer.trainer import Trainer


class RepresentativesTrainer(Trainer):
    """
    Trains with the proposed algorithm that uses RMSProp and representatives selection for incremental learning
    """

    def __init__(self, config: GeneralConfig, model: Network, pipeline: Data, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                 tester: Tester = None, checkpoint: str = None):
        super().__init__(config, model, pipeline, tensor_x, tensor_y, tester=tester, checkpoint=checkpoint)

        self.representatives = [[] for _ in range(model.get_output().shape[1])]
        self.class_count = [0 for _ in range(model.get_output().shape[1])]
        self.weights = tf.placeholder(tf.float32, [None])

        self.buffered_reps = []
        self.buffer = 1  # Number of buffer iterations. Interval at which the representatives will be updated

        # TODO temporal placeholders for tests
        test = 1
        # Test for random with minimum values (1%)
        if test == 1:
            self.random = True
            if self.config.model_name == 'FASHION-MNIST':
                self.presel = 40  # Number of preselected samples
                self.rep_per_batch = 40  # Number of representatives that are passed in each batch
                self.rep_per_class = 50  # Maximum number of representatives per class
            elif self.config.model_name == 'CALTECH-101':
                self.presel = 20  # Number of preselected samples
                self.rep_per_batch = 20  # Number of representatives that are passed in each batch
                self.rep_per_class = 1  # Maximum number of representatives per class
            else:
                self.presel = 20  # Number of preselected samples
                self.rep_per_batch = 20  # Number of representatives that are passed in each batch
                self.rep_per_class = 50  # Maximum number of representatives per class

        # Tests for random maximum values (10%)
        elif test == 2:
            self.random = True
            if self.config.model_name == 'FASHION-MNIST':
                self.presel = 40  # Number of preselected samples
                self.rep_per_batch = 40  # Number of representatives that are passed in each batch
                self.rep_per_class = 500  # Maximum number of representatives per class
            elif self.config.model_name == 'CALTECH-101':
                self.presel = 20  # Number of preselected samples
                self.rep_per_batch = 20  # Number of representatives that are passed in each batch
                self.rep_per_class = 10  # Maximum number of representatives per class
            else:
                self.presel = 20  # Number of preselected samples
                self.rep_per_batch = 20  # Number of representatives that are passed in each batch
                self.rep_per_class = 500  # Maximum number of representatives per class

        else:
            self.random = False
            self.presel = 20  # Number of preselected samples
            self.rep_per_batch = 20  # Number of representatives that are passed in each batch
            self.rep_per_class = 50  # Maximum number of representatives per class

    def _create_loss(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        return tf.losses.softmax_cross_entropy(tf.multiply(tensor_y, self.mask_tensor),
                                               tf.multiply(net_output, self.mask_tensor), weights=self.weights)

    def _create_optimizer(self, config: GeneralConfig, loss: tf.Tensor, var_list=None):
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
        if self.random:
            if n_reps == 0:
                self.__random_buffer(image_batch, target_batch, outputs, total_it, megabatch)
            else:
                self.__random_buffer(image_batch[:-n_reps], target_batch[:-n_reps], outputs[:-n_reps], total_it,
                                     megabatch)
            if total_it % self.buffer == 0:
                self.__random_modify_representatives(self.buffered_reps)
                self.__clear_buffer()
        else:
            if n_reps == 0:
                self.__buffer_samples(image_batch, target_batch, outputs, total_it, megabatch)
            else:
                self.__buffer_samples(image_batch[:-n_reps], target_batch[:-n_reps], outputs[:-n_reps], total_it,
                                      megabatch)
            if total_it % self.buffer == 0:
                self.__modify_representatives(self.buffered_reps)
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
            samples = np.random.choice(repr_list, size=min(self.rep_per_batch, repr_list.size), replace=False)
            return samples
        else:
            return []

    def __buffer_samples(self, image_batch, target_batch, outputs, iteration, megabatch):
        """
        Adds samples to the buffer. This version buffers all the original images from a batch

        :param image_batch: the list of images of a batch
        :param target_batch: the list of one hot labels of a batch
        :param outputs: output probabilities of the neural network
        :param iteration: current iteration of training
        :param megabatch: current megabatch

        :return: None
        """
        scores_ranking = np.argsort(outputs)  # Allows knowing the ranking of each class probability
        sorted_outputs = np.sort(outputs)  # Aux. for outputs sorted by probability
        difs = np.array([i[-1] - i[-2] for i in sorted_outputs])  # Best vs. Second Best
        sort_indices = np.argsort(difs)  # Order indices (from lowest dif. to highest dif.)
        difs = difs[sort_indices]
        image_batch = np.asarray(image_batch)[sort_indices]  # The data is ordered according to the indices
        target_batch = np.asarray(target_batch)[sort_indices]  # The data labels are ordered according to the indices
        scores_ranking = scores_ranking[sort_indices]
        outputs = outputs[sort_indices]

        x = 0
        initial_index = max(math.floor(len(image_batch) / 2 - self.presel / 2) - x, 0)
        end_index = min(math.floor(len(image_batch) / 2 + self.presel / 2) - x, len(image_batch))

        for i in range(initial_index, end_index):
            self.buffered_reps.append(
                Representative(image_batch[i].copy(), target_batch[i].copy(), difs[i].copy(), iteration, megabatch,
                               outputs[i].copy()))

    def __modify_representatives(self, candidate_representatives):
        """
        Modifies the representatives list according to the new data
        
        :param candidate_representatives: the preselected representatives from the buffer
        :return: None
        """
        for i, _ in enumerate(candidate_representatives):
            nclass = int(np.argmax(candidate_representatives[i].label))
            self.representatives[nclass].append(candidate_representatives[i])
            self.class_count[nclass] += 1

        # self.__recalculate_metrics(self.representatives)
        # Sorts representatives of each list, corresponding to each class
        self.__calculate_crowd_distance(self.representatives)
        for i in range(len(self.representatives)):
            self.representatives[i].sort(key=lambda x: x.crowd_distance)
            # self.representatives[i].sort(key=lambda x: x.metric * (1 + min(1, (total_it - x.iteration) / 10000)),
            #                             reverse=True)
            self.representatives[i] = self.representatives[i][-min(self.rep_per_class, len(self.representatives[i])):]

        self.__recalculate_weights(self.representatives)

    def __random_buffer(self, image_batch, target_batch, outputs, iteration, megabatch):
        """
        Creates a buffer based in random sampling

        :param image_batch: the list of images of a batch
        :param target_batch: the list of one hot labels of a batch
        :param outputs: output probabilities of the neural network
        :param iteration: current iteration of training
        :param megabatch: current megabatch
        :return: None
        """
        rand_indices = np.random.permutation(len(outputs))
        outputs = outputs[rand_indices]
        difs = [0 for _ in outputs]
        image_batch = np.asarray(image_batch)[rand_indices]  # The data is ordered according to the indices
        target_batch = np.asarray(target_batch)[rand_indices]
        for i in range(min(self.presel, len(image_batch))):
            self.buffered_reps.append(
                Representative(image_batch[i].copy(), target_batch[i].copy(), difs[i], iteration, megabatch,
                               outputs[i].copy()))

    def __random_modify_representatives(self, candidate_representatives):
        """
            Modifies the representatives list according to the new data by selecting representatives randomly from the
            buffer and the current list of representatives

            param candidate_representatives: the preselected representatives from the buffer
            :return: None
        """
        for i, _ in enumerate(candidate_representatives):
            nclass = int(np.argmax(candidate_representatives[i].label))
            self.representatives[nclass].append(candidate_representatives[i])
            self.class_count[nclass] += 1

        for i in range(len(self.representatives)):
            rand_indices = np.random.permutation(len(self.representatives[i]))
            self.representatives[i] = [self.representatives[i][j] for j in rand_indices]
            # self.representatives[i].sort(key=lambda x: x.metric * (1 + min(1, (total_it - x.iteration) / 10000)),
            #                             reverse=True)
            self.representatives[i] = self.representatives[i][-min(self.rep_per_class, len(self.representatives[i])):]

        self.__recalculate_weights(self.representatives)

    def __clear_buffer(self):
        """
        Clears the buffer
        :return: None
        """
        self.buffered_reps = []

    # TODO hacer más eficiente
    @staticmethod
    def __calculate_crowd_distance(representatives):
        """
        Calculates or recalculates the crowd distances for all the representatives

        :param representatives: the list of representatives divided by class
        :return: None
        """
        # Resets crowd distance for everyone
        for cls in representatives:
            for rep in cls:
                rep.crowd_distance = 0

        # Recalculates crowd distance for everyone
        for cls in representatives:
            # Pass if the class doesn't have any samples
            if len(cls) == 0:
                continue
            # Travels each 'objective'
            # TODO: no necesariamente clase REAL == Best. Es posible que la red clasifique mal al representante
            for i in range(len(cls[0].net_output)):

                def sort(x):
                    return x.net_output[i]

                cls.sort(key=sort)
                # Calculates crowd distance for the i-th objective.
                # cls[0].crowd_distance, cls[-1].crowd_distance = math.inf, math.inf
                for j in range(1, len(cls) - 1):
                    cls[j].crowd_distance += cls[j].net_output[i] - cls[j - 1].net_output[i]
                    cls[j].crowd_distance += cls[j + 1].net_output[i] - cls[j].net_output[i]

    def __recalculate_weights(self, representatives):
        """
        Reassigns the weights of the representatives
        :param representatives: a list of representatives
        :return: None
        """
        total_count = np.sum(self.class_count)
        # This version proposes that the total weight of representatives is calculated from the proportion of candidate
        # representatives respect to the batch. E.g. a batch of 100 images and 10 are preselected, total_weight = 10
        total_weight = (self.config.train_configurations[0].batch_size * 1.0) / self.presel
        # The total_weight is adjusted to the proportion between candidate representatives and actual representatives
        total_weight *= (total_count / np.sum([len(cls) for cls in representatives]))
        probs = [count / total_count for count in self.class_count]
        for i in range(len(representatives)):
            for rep in representatives[i]:
                # This version uses log as an stabilizer
                rep.weight = max(math.log(probs[i] * total_weight), 1.0)

    def __recalculate_metrics(self, representatives):
        """
        Reassigns the metrics of the representatives
        :param representatives: a list of representatives
        :return: None
        """
        aux_rep = [rep for cls in representatives for rep in cls]
        image_batch = np.array([rep.value for rep in aux_rep])
        outputs = self.sess.run(self.model.get_output(), feed_dict={self.tensor_x: image_batch})
        scores_ranking = np.argsort(outputs)  # Allows knowing the ranking of each class probability
        aux_outputs = np.sort(outputs)  # Order outputs over the last axis
        difs = np.array([i[-1] - i[-2] for i in aux_outputs])  # Best vs. Second Best
        for i in range(len(aux_rep)):
            aux_rep[i].metric = difs[i]
            aux_rep[i].net_output = outputs[i]


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
