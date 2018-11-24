"""
Module for performing testing/validation of a model
"""
import os
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from input.data import Data
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
        self.streaming_accuracy, self.streaming_accuracy_update, self.streaming_accuracy_scalar = None, None, None
        self.test_iterator = None
        self.test_x, self.test_y = None, None
        pass

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
        # Creates variables for measuring the accuracy
        correct_prediction = tf.equal(tf.argmax(self.tensor_y, 1), tf.argmax(self.model.get_output(), 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.streaming_accuracy, self.streaming_accuracy_update = tf.metrics.mean(accuracy)
        self.streaming_accuracy_scalar = tf.summary.scalar('accuracy', self.streaming_accuracy)

        self.test_iterator, self.test_x, self.test_y = self.pipeline.build_test_data_tensor()

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
        sess.run(self.test_iterator.initializer)
        sess.run(tf.variables_initializer(tf.get_default_graph().get_collection(tf.GraphKeys.METRIC_VARIABLES)))
        while True:
            try:
                test_images, test_target = sess.run([self.test_x, self.test_y])
                self.streaming_accuracy = sess.run([self.streaming_accuracy_update],
                                                   feed_dict={self.tensor_x: test_images,
                                                              self.tensor_y: test_target,
                                                              self.model.use_dropout: 0.0})
            except OutOfRangeError:
                print("Finished validation of iteration {}...".format(iteration))
                break

        summary = sess.run(self.streaming_accuracy_scalar)
        writer.add_summary(summary, iteration)
