"""
Module for training a neural network
"""
import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from input.data import Data
from network import Network
from training.train_conf import GeneralConfig
import utils.dir_utils as utils


class Trainer(object):
    """
    Has the purpose of training a generic net, with a generic configuration and optimizer
    """

    def __init__(self, config: GeneralConfig, model: Network, pipeline: Data, tensor_x: tf.Tensor, tensor_y: tf.Tensor):
        """
        It creates a Trainer object
        :param config: the configuration for the whole training
        :param model: the neural net that is going to be trained
        :param pipeline: the data pipeline for the training
        """
        self.config = config
        self.model = model
        self.pipeline = pipeline
        self.tensor_x = tensor_x
        self.tensor_y = tensor_y

        # Creation of aditional attributes for training
        self.sess = None
        self.writer = None
        self.ck_path, self.summaries_path = None, None
        self.streaming_accuracy_update, self.streaming_accuracy_scalar = None, None
        self.train_step = None

    def __prepare(self):
        """
        It does the preparation for the training
        :return: None
        """
        print("Preparing training...")

        self.ck_path, self.summaries_path = utils.prepare_directories(self.config)

        # Creates the session
        sess = tf.get_default_session()
        if sess:
            sess.close()
        self.sess = tf.InteractiveSession()

        # Creates mse and summaries
        self.mse = tf.reduce_mean(tf.square(self.tensor_y - self.model.get_output()))
        correct_prediction = tf.equal(tf.argmax(self.tensor_y, 1), tf.argmax(self.model.get_output(), 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        streaming_accuracy, self.streaming_accuracy_update = tf.metrics.mean(accuracy)
        self.streaming_accuracy_scalar = tf.summary.scalar('accuracy', streaming_accuracy)

        # TODO cambiar para poder crear un Optimizer genérico
        self.train_step = tf.train.RMSPropOptimizer(0.01).minimize(self.mse)

        # TODO carga de checkpoints
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        print("Finished preparations for training...")

    # TODO hacerlo genérico para cualquier Optimizer
    # TODO si hay checkpoint, cómo se va a manejar el restablecimiento del batch correcto de datos? Es decir, se debe reestablecer el mega batch correcto
    def train(self):
        """
        Trains a neural network with the appropriate configuration.
        :return: None
        """
        self.__prepare()

        self.writer = tf.summary.FileWriter(self.summaries_path, tf.get_default_graph())
        test_x, test_y = self.pipeline.build_test_data_tensor()

        for i, _ in enumerate(self.config.train_configurations):
            data_x, data_y = self.pipeline.build_train_data_tensor()
            self.train_increment(self.config, self.writer, data_x, data_y, test_x, test_y)
            print("Finished training of increment {}...".format(i))
            if i + 1 < len(self.config.train_configurations):
                self.pipeline.change_dataset_part(i + 1)

        self.__finish()

    # TODO mensajes con porcentajes de avance
    def train_increment(self, config: GeneralConfig, writer: tf.summary.FileWriter,
                        data_x: tf.Tensor, data_y: tf.Tensor,
                        test_x: tf.Tensor, test_y: tf.Tensor):
        """
        Trains a neural network over an increment of data, with the given configuration and model, over the data that is
        provided. It also saves summaries for TensorBoard and creates checkpoints according to the given configuration.
        :param config: the configuration for the whole training
        :param writer: a FileWriter properly configured
        :param data_x: the tensor associated with the training data
        :param data_y: the tensor that has the corresponding labels of the training data
        :param test_x: the tensor associated with the testing/validation data
        :param test_y: the tensor that has the corresponding labels of the testing/validation data
        :return: None
        """
        i = 0
        while True:
            try:
                image_batch, target_batch = self.sess.run([data_x, data_y])
                _, c = self.sess.run([self.train_step, self.mse],
                                     feed_dict={self.tensor_x: image_batch, self.tensor_y: target_batch})
                if i % config.summary_interval == 0:
                    print("Performing validation at iteration number: {}. Mse is: {}".format(i, c))
                    self.__perform_validation(i, writer, test_x, test_y)

            except OutOfRangeError:
                break
            i += 1

    def __perform_validation(self, iteration: int, writer: tf.summary.FileWriter, test_x: tf.Tensor, test_y: tf.Tensor):
        """
        Performs validation over the test data and register the results in the form of summaries that can be interpreted
        by Tensorboard
        :type iteration: the current iteration number over the training data
        :type writer: a FileWriter properly configured
        :param test_x: the tensor associated with the testing/validation data
        :param test_y: the tensor that has the corresponding labels of the data
        :return: None
        """
        while True:
            try:
                test_images, test_target = self.sess.run([test_x, test_y])
                self.sess.run([self.streaming_accuracy_update],
                              feed_dict={self.tensor_x: test_images, self.tensor_y: test_target})
            except OutOfRangeError:
                print("Finished validation of iteration {}...".format(iteration))
                break

        summary = self.sess.run(self.streaming_accuracy_scalar)
        writer.add_summary(summary, iteration)

    def __finish(self):
        """
        Performs the required operations to finish the training (like closing pipelines)
        :return: None
        """
        print("Finishing training...")
        self.pipeline.close()
        self.writer.close()
