"""
Module for performing testing/validation of a model
"""
import os
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError
from tensorflow.python.framework import ops

from etl.data import Data
from libs.caffe_tensorflow.network import Network


class Tester(object):
    """
    Class for performing and saving tests of a model, with various metrics (e.g. accuracy)
    """

    def __init__(self, model: Network, pipeline: Data, tensor_x: tf.Tensor, tensor_y: tf.Tensor):
        """
        It creates a Tester object

        :param model: the neural net that is going to be trained
        :param pipeline: the data pipeline for the training
        :param tensor_x: the tensor corresponding to the input of a training
        :param tensor_y: the tensor corresponding to the output of a trainings

        This method must be called by the constructors of the subclasses.
        """
        self.model = model
        self.pipeline = pipeline
        self.tensor_x = tensor_x
        self.tensor_y = tensor_y

        self.writer = None
        self.test_iterator = None
        self.test_x, self.test_y = None, None
        self.scalar_tensors, self.update_tensors = list(), list()
        self.aux_tensors = dict()

    @staticmethod
    def create_writer(summaries_path: str, identifier: int):
        """
        Creates a tf.summary.FileWriter, that saves results to summaries_path/increment_{id}

        :param summaries_path: the path of the directory where the results of the tests are going to be saved
        :param identifier: an ID for identifying the writer. This is used for the name of the file where the results are
            going to be saved
        :return: a tf.summary.FileWriter object
        """
        return tf.summary.FileWriter(os.path.join(summaries_path, "increment_{}".format(identifier)),
                                     tf.get_default_graph())

    def prepare(self):
        """
        It does the preparation for the training. This preparations include:
        -Creates operators needed for summaries for TensorBoard

        :return: None
        """
        self._create_aux_tensors()
        self._create_accuracy_metric()
        self._create_precision_metric()
        self._create_recall_metric()
        self._create_fscore_metric()
        self._create_gmean_metric()
        self._create_loss_metric()

        self.test_iterator, self.test_x, self.test_y = self.pipeline.build_test_data_tensor()

    def _create_aux_tensors(self):
        """
        Creates the auxiliary tensors for True Positives (TP), False Negatives (FN) and False Positives (FP).
        This method should be used before any of the other metrics is created, as those methods may
        require a tensor that is created here.
        The tensors are added to the aux_tensors dictionary

        :return: None
        """
        num_classes = tf.shape(self.tensor_y)[1]
        one_hot_preds = tf.one_hot(tf.argmax(self.model.get_output(), 1), num_classes)
        correct_prediction = tf.logical_and(tf.cast(self.tensor_y, tf.bool), tf.cast(one_hot_preds, tf.bool))
        true_positives = tf.reduce_sum(tf.cast(correct_prediction, tf.float32), 0)
        false_positives = tf.subtract(tf.reduce_sum(one_hot_preds, 0), true_positives)
        false_negatives = tf.subtract(tf.reduce_sum(self.tensor_y, 0), true_positives)
        tp_total, fp_total, fn_total = self._create_totals(true_positives, false_positives, false_negatives)

        self.aux_tensors["TP_TOTAL"] = tp_total
        self.aux_tensors["FP_TOTAL"] = fp_total
        self.aux_tensors["FN_TOTAL"] = fn_total

    def _create_totals(self, true_positives: tf.Tensor, false_positives: tf.Tensor,
                       false_negatives: tf.Tensor):
        """
        Creates the tensors to store the total True Positives, False Positives and False Negatives

        :param true_positives: the tensor with the true positives count per class for each batch
        :param false_positives: the tensor with the false positives count per class for each batch
        :param false_negatives: the tensor with the false positives count per class for each batch
        :return: Three tensors that contain the total counts
        """
        tp_mac = tf.Variable(tf.zeros(self.model.get_output().shape[1], tf.float32), trainable=False,
                             validate_shape=False,
                             collections=[ops.GraphKeys.METRIC_VARIABLES], name="total_true_positives")
        fp_mac = tf.Variable(tf.zeros(self.model.get_output().shape[1], tf.float32), trainable=False,
                             validate_shape=False,
                             collections=[ops.GraphKeys.METRIC_VARIABLES], name="total_false_positives")
        fn_mac = tf.Variable(tf.zeros(self.model.get_output().shape[1], tf.float32), trainable=False,
                             validate_shape=False,
                             collections=[ops.GraphKeys.METRIC_VARIABLES], name="total_false_negatives")
        up_tp_mac = tf.assign_add(tp_mac, true_positives)
        up_fp_mac = tf.assign_add(fp_mac, false_positives)
        up_fn_mac = tf.assign_add(fn_mac, false_negatives)
        self.update_tensors.append(up_tp_mac)
        self.update_tensors.append(up_fp_mac)
        self.update_tensors.append(up_fn_mac)
        return tp_mac, fp_mac, fn_mac

    def _create_accuracy_metric(self):
        """
        Creates the accuracy metric and its auxiliary tensors and adds them to the local tester graph

        :return: None
        """
        streaming_accuracy, streaming_accuracy_update = tf.metrics.accuracy(tf.argmax(self.tensor_y, 1),
                                                                            tf.argmax(self.model.get_output(), 1),
                                                                            name='accuracy_metric')
        self.scalar_tensors.append(tf.summary.scalar('accuracy', streaming_accuracy))
        self.update_tensors.append(streaming_accuracy_update)

    def _create_precision_metric(self):
        """
        Creates the precision metric and its auxiliary tensors and adds them to the local tester graph

        :return: None
        """
        tp_total = self.aux_tensors["TP_TOTAL"]
        fp_total = self.aux_tensors["FP_TOTAL"]
        precision = tf.divide(tp_total, self.__add_non_zero(tp_total, fp_total))
        precision_scalar = tf.reduce_mean(precision)

        self.scalar_tensors.append(tf.summary.scalar('precision', precision_scalar))
        self.aux_tensors["PRECISION"] = precision

    def _create_recall_metric(self):
        """
        Creates the recall metric and its auxiliary tensors and adds them to the local tester graph

        :return: None
        """
        tp_total = self.aux_tensors["TP_TOTAL"]
        fn_total = self.aux_tensors["FN_TOTAL"]
        recall = tf.divide(tp_total, self.__add_non_zero(tp_total, fn_total))
        recall_scalar = tf.reduce_mean(recall)

        self.scalar_tensors.append(tf.summary.scalar('recall', recall_scalar))
        self.aux_tensors["RECALL"] = recall

    def _create_fscore_metric(self):
        """
        Creates the F-score metric and its auxiliary tensors and adds them to the local tester graph.
        The Precision and Recall metrics must already be created

        :return: None
        """
        precision = self.aux_tensors["PRECISION"]
        recall = self.aux_tensors["RECALL"]
        fscore = tf.divide(tf.multiply(precision, recall), self.__add_non_zero(precision, recall))
        fscore_scalar = tf.reduce_mean(fscore) * 2
        self.scalar_tensors.append(tf.summary.scalar('fscore', fscore_scalar))

    def _create_gmean_metric(self):
        """
        Creates the G-Mean metric and its auxiliary tensors and adds them to the local tester graph.
        The Recall metric must already be created

        :return: None
        """
        recall = self.aux_tensors["RECALL"]
        gmean_scalar = tf.pow(tf.reduce_prod(recall), tf.cast(1/tf.shape(recall)[0], tf.float32))
        self.scalar_tensors.append(tf.summary.scalar('gmean', gmean_scalar))

    @staticmethod
    def _create_loss_metric():
        """
        Creates the loss metric and its auxiliary tensors

        :return: None
        """
        loss_tensor = tf.placeholder(dtype=tf.float32, shape=(), name='loss_tensor')
        tf.summary.scalar('loss', loss_tensor)

    def perform_validation(self, sess, iteration: int, writer: tf.summary.FileWriter):
        """
        Performs validation over the test data and register the results in the form of summaries that can be interpreted
        by Tensorboard. The prepare method must have been called at least once before using this method, otherwise,
        an Exception may occur.

        :param sess: the current session
        :param iteration: the current iteration number over the training data
        :param writer: a FileWriter properly configured
        :return: None
        """
        update_tensors = self.update_tensors
        scalar_tensors = self.scalar_tensors

        sess.run(self.test_iterator.initializer)
        sess.run(tf.variables_initializer(tf.get_default_graph().get_collection(tf.GraphKeys.METRIC_VARIABLES)))
        while True:
            try:
                test_images, test_target = sess.run([self.test_x, self.test_y])
                sess.run(update_tensors,
                         feed_dict={self.tensor_x: test_images,
                                    self.tensor_y: test_target,
                                    self.model.use_dropout: 0.0})
            except OutOfRangeError:
                print("Finished validation of iteration {}...".format(iteration))
                break

        for tensor in scalar_tensors:
            summary = sess.run(tensor)
            writer.add_summary(summary, iteration)

    @staticmethod
    def save_loss(sess, loss, iteration: int, writer: tf.summary.FileWriter):
        """
        Saves the loss value into the summary file for TensorBoard

        :param sess: the current session
        :param loss: the loss value for the current iteration
        :param iteration: the current iteration number over the training data
        :param writer: a FileWriter properly configured
        :return: None
        """
        loss_tensor = tf.get_default_graph().get_tensor_by_name('loss_tensor:0')
        loss_scalar = tf.get_default_graph().get_tensor_by_name('loss:0')
        summary = sess.run(loss_scalar, feed_dict={loss_tensor: loss})
        writer.add_summary(summary, iteration)

    def __add_non_zero(self, first_tensor: tf.Tensor, second_tensor: tf.Tensor):
        """
        Adds two tensors and an additional auxiliary tensor to make sure that the resulting
        value is always greater than zer if both first_tensor and second_tensor contain only positive values

        :param first_tensor: the first tensor to be added
        :param second_tensor: the second tensor to be added
        :return: a tensor with the sum of first_tensor and second_tensor
        """
        return tf.add(tf.add(first_tensor, second_tensor),
                      tf.fill(tf.shape(first_tensor), 0.001))
