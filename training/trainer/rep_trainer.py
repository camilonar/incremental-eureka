"""
The proposed algorithm that uses RMSProp and representatives selection for incremental learning
"""
import math
import tensorflow as tf
import numpy as np

from input.data import Data
from libs.caffe_tensorflow.network import Network
from training.config.general_config import GeneralConfig
from training.support.tester import Tester
from training.trainer.trainer import Trainer


class RepresentativesTrainer(Trainer):
    """
    Trains with the proposed algorithm that uses RMSProp and representatives selection for incremental learning
    """

    def __init__(self, config: GeneralConfig, model: Network, pipeline: Data, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                 tester: Tester = None, checkpoint: str = None):
        super().__init__(config, model, pipeline, tensor_x, tensor_y, tester=tester, checkpoint=checkpoint)

        self.representatives = [[] for _ in range(model.get_output().shape[1])]
        self.weights = tf.placeholder(tf.float32, [None])

        self.presel = 10  # Number of preselected samples
        self.n = 10  # Number of representatives that are passed in each batch
        self.r = 20  # Maximum number of representatives per class

    def _create_loss(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        return tf.losses.softmax_cross_entropy(tensor_y, net_output, weights=self.weights)

    def _create_optimizer(self, config: GeneralConfig, loss: tf.Tensor, var_list=None):
        return tf.train.RMSPropOptimizer(config.learn_rate).minimize(loss, var_list=var_list)

    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, loss: tf.Tensor, increment: int, iteration: int, total_it: int):
        # Gets the representatives
        rep_values, rep_labels = self.__get_representatives(sess, image_batch, target_batch,
                                                            tensor_x, tensor_y, total_it)

        # Gets the respective weights
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

    def __get_representatives(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                              total_it: int):
        """
        Selects or retrieves the representatives from the data

        :param sess: the current session
        :param image_batch: he batch of data corresponding to the input, as obtained from the data pipeline
        :param target_batch: the batch of data corresponding to the output, as obtained from the data pipeline
        :param tensor_x: the tensor corresponding to the input of a training
        :param tensor_y: the tensor corresponding to the output of a training
        :param total_it: the current iteration counting from the start of training
        :return: a tuple with 2 numpy.ndarray with the data and the labels. The data array has shape
            **[n_representatives, x1, x2, ..., xn]** where [x1...xn] is the shape of a single sample image. The labels
            array has shape **[n_representatives, n_labels]**.
        """
        outputs = sess.run(self.model.get_output(), feed_dict={tensor_x: image_batch})

        scores_ranking = np.argsort(outputs)  # Allows knowing the ranking of each class probability
        outputs = np.sort(outputs)  # Order outputs over the last axis
        difs = [i[-1] - i[-2] for i in outputs]  # Best vs. Second Best
        sort_indices = np.argsort(difs)  # Order indices (from lowest dif. to highest dif.)
        difs = [difs[i] for i in sort_indices]
        image_batch = [image_batch[i] for i in sort_indices]  # The data is ordered according to the indices
        target_batch = [target_batch[i] for i in sort_indices]  # The data labels are ordered according to the indices
        scores_ranking = [scores_ranking[i] for i in sort_indices]
        outputs = [outputs[i] for i in sort_indices]

        self.__modify_representatives(image_batch[:self.presel], target_batch[:self.presel], difs[:self.presel],
                                      outputs, scores_ranking, total_it)
        samples = np.random.choice(np.concatenate(self.representatives), size=self.n, replace=False)
        return [i.value for i in samples], [i.output for i in samples]

    def __modify_representatives(self, preselected_images, preselected_targets, metric_values, sorted_scores,
                                 scores_ranking, total_it: int):
        """
        Modifies the representatives list according to the new data
        
        :param preselected_images: the preselected representatives from the current batch
        :param preselected_targets: the preselected representatives' labels
        :param metric_values: the metric values of each one of the preselected representatives
        :param sorted_scores: the sorted output of the network for preselected samples. Must have shape
                [n_samples, n_outputs]. The scores must be sorted by axis=-1
        :param scores_ranking: the position of each class probability of each one of the preselected representatives.
                I.e. an array of shape [n_samples, n_outputs], where n_outputs has the values as:
                [last_class_index,..., second_class_index, first_class_index]
                E.g. if we have two samples where the outputs for the network are [0.3, 0.2, 0.5] and [0.7, 0.5, 0.6],
                then output_ranking should be: [[1, 0, 2], [1, 2, 0]]
        :param total_it: the current iteration counting from the start of training
        :return:
        """
        for i, _ in enumerate(preselected_images):
            nclass = int(np.argmax(preselected_targets[i]))
            self.representatives[nclass].append(Representative(preselected_images[i], preselected_targets[i],
                                                               metric_values[i], total_it,
                                                               self.__extract_best_second_best(sorted_scores[i],
                                                                                               scores_ranking[i])))

        # Sorts representatives of each list, corresponding to each class
        self._calculate_crowd_distance(self.representatives)
        for i in range(len(self.representatives)):
            # self.representatives[i].sort(key=lambda x: x.crowd_distance)
            self.representatives[i].sort(key=lambda x: x.metric * (1 + min(1, (total_it - x.iteration) / 10000)),
                                         reverse=True)
            self.representatives[i] = self.representatives[i][-min(self.r, len(self.representatives[i])):]

    @staticmethod
    def __extract_best_second_best(scores, scores_ranking):
        """
        Extracts the best and second best

        :param scores: the sorted output of the network for a sample. Must have shape [n_outputs].
                The scores must be sorted by axis=-1
        :param scores_ranking: the position of each class probability of each one of the preselected representatives.
                I.e. an array of shape [n_outputs], with the following structure:
                [last_class_index,..., second_class_index, first_class_index]
                E.g. if we have a samples where the outputs for the network are [0.3, 0.2, 0.5], then output_ranking
                should be: [1, 0, 2]
        :return: a tuple of tuples, with the structure:
                ((first_class_index, first_class_score), (second_class_index, second_class_score))
        """
        bvsb = ((scores_ranking[-1], scores[-1]), (scores_ranking[-2], scores[-2]))
        return bvsb

    # TODO hacer más eficiente
    def _calculate_crowd_distance(self, representatives):
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
            for i in range(len(cls[0].reduced_rep)):

                def sort(x):
                    return x.reduced_rep[i][1]

                cls.sort(key=sort)
                # Calculates crowd distance for the i-th objective.
                cls[0].crowd_distance, cls[-1].crowd_distance = math.inf, math.inf
                for j in range(1, len(cls) - 1):
                    cls[j].crowd_distance += cls[j].reduced_rep[i][1] - cls[j - 1].reduced_rep[i][1]
                    cls[j].crowd_distance += cls[j + 1].reduced_rep[i][1] - cls[j].reduced_rep[i][1]


class Representative(object):
    """
    Representative sample of the algorithm
    """

    def __init__(self, value, output, metric, iteration, reduced_rep=None, crowd_distance=None):
        """
        Creates a Representative object
        :param value: the value of the representative (i.e. the image)
        :param output: the expected ground truth label (in one-hot format)
        :param metric: the value of the metric
        :param iteration: the iteration at which the sample was selected as representative
        :param reduced_rep: some kind of reduced representation of the representative (e.g. Best and Second Best class)
        :param crowd_distance: a measure of distance to the other representatives of the same cluster (e.g. same class)
        """
        self.value = value
        self.output = output
        self.metric = metric
        self.iteration = iteration
        self.reduced_rep = reduced_rep
        self.crowd_distance = crowd_distance
