"""
The proposed algorithm that uses RMSProp and representatives selection for incremental learning
"""
import random
import tensorflow as tf
import numpy as np

from input.data import Data
from libs.caffe_tensorflow.network import Network
from training.config.general_config import GeneralConfig
from training.support.tester import Tester
from training.trainer.trainer import Trainer


# TODO implementar
class RepresentativesTrainer(Trainer):
    """
    Trains with the proposed algorithm that uses RMSProp and representatives selection for incremental learning
    """

    def __init__(self, config: GeneralConfig, model: Network, pipeline: Data, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                 tester: Tester = None, checkpoint: str = None):
        super().__init__(config, model, pipeline, tensor_x, tensor_y, tester=tester, checkpoint=checkpoint)
        self.representatives = []
        self.weights = tf.placeholder(tf.float32, [None])

    def _create_loss(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        return tf.losses.softmax_cross_entropy(tensor_y, net_output, weights=self.weights)

    def _create_optimizer(self, config: GeneralConfig, loss: tf.Tensor, var_list=None):
        return tf.train.RMSPropOptimizer(config.learn_rate).minimize(loss, var_list=var_list)

    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, loss: tf.Tensor, increment: int, iteration: int, total_it: int):
        # Gets the representatives
        rep_values, rep_labels = self.__get_representatives(sess, image_batch, target_batch,
                                                            tensor_x, tensor_y, total_it)

        # Gets the respectives weights
        weights_values = np.full((len(image_batch)), 1.0)
        rep_weights = np.full((len(rep_values)), 3.0)
        if increment == 0:
            weights_values = np.concatenate((weights_values, rep_weights))

            # Concatenates the training samples with the representatives
            image_batch = np.concatenate((image_batch, rep_values))
            target_batch = np.concatenate((target_batch, rep_labels))
            # Executes the update of the net
            return self.sess.run([self.train_step, self.loss],
                                 feed_dict={self.tensor_x: image_batch, self.tensor_y: target_batch,
                                            self.weights: weights_values})
        else:
            return self.sess.run([self.train_step, self.loss],
                                 feed_dict={self.tensor_x: image_batch, self.tensor_y: target_batch,
                                            self.weights: weights_values})

    # TODO implementar. Por ahora retorna representantes aleatorios. Evaluar: uso de fronteras y centros
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
        n = 10  # Number of representatives
        outputs = sess.run(self.model.get_output(), feed_dict={tensor_x: image_batch, tensor_y: target_batch})

        outputs = np.sort(outputs)  # Order outputs over the last axis
        difs = [i[-1] - i[-2] for i in outputs]  # Best vs. Second Best
        sort_indices = np.argsort(difs)  # Order indices (from lowest dif. to highest dif.)
        difs = [difs[i] for i in sort_indices]
        image_batch = [image_batch[i] for i in sort_indices]  # The data is ordered according to the indices
        target_batch = [target_batch[i] for i in sort_indices]  # The data labels are ordered according to the indices

        self.__modify_representatives(image_batch[:n], target_batch[:n], difs[:n], total_it)
        samples = random.sample(self.representatives, n)
        return [i.value for i in samples], [i.output for i in samples]

    def __modify_representatives(self, preselected_images, preselected_targets, metric_values, total_it: int):
        """
        Modifies the representatives list according to the new data
        
        :param preselected_images: the preselected representatives from the current batch
        :param preselected_targets: the preselected representatives' labels
        :param metric_values: the metric values of each one of the preselected representatives
        :param total_it: the current iteration counting from the start of training
        :return:
        """
        r = 200  # Number of total representatives
        for i, _ in enumerate(preselected_images):
            self.representatives.append(Representative(preselected_images[i], preselected_targets[i], metric_values[i],
                                                       total_it))
            self.representatives.sort(key=lambda x: x.metric * (1 + min(1, (total_it - x.iteration)/10000)),
                                      reverse=True)
            self.representatives = self.representatives[-min(r, len(self.representatives)):]


class Representative(object):
    """
    Representative sample of the algorithm
    """
    def __init__(self, value, output, metric, iteration):
        """
        Creates a Representative object
        :param value: the value of the representative (i.e. the image)
        :param output: the expected ground truth label (in one-hot format)
        :param metric: the value of the metric
        :param iteration: the iteration at which the sample was selected as representative
        """
        self.value = value
        self.output = output
        self.metric = metric
        self.iteration = iteration
