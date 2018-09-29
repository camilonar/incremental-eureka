"""
Module for training a neural network.
Features:
1. It's independent of the dataset, or model used for training
2. Does all the needed preparations for the training (e.g. creating a session)
"""
from collections import Counter

import numpy
import os
import time
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from input.data import Data
from network import Network
from training.train_conf import GeneralConfig
import utils.dir_utils as utils


class Trainer(ABC):
    """
    Has the purpose of training a generic net, with a generic configuration. It serves as a base for custom trainers
    """

    def __init__(self, config: GeneralConfig, model: Network, pipeline: Data, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                 checkpoint: str = None):
        """
        It creates a Trainer object
        :param config: the configuration for the whole training
        :param model: the neural net that is going to be trained
        :param pipeline: the data pipeline for the training
        :param checkpoint: the checkpoint path if it's required to start the training from a checkpoint. A data path
        with the following structure is expected: ./checkpoints/dataset_name/config_net_name/checkpoint_name.ckpt.
        If there is no checkpoint to be loaded then its value should be None. The default value is None.

        This method must be called by the constructors of the subclasses.
        """
        self.config = config
        self.model = model
        self.pipeline = pipeline
        self.tensor_x = tensor_x
        self.tensor_y = tensor_y
        self.checkpoint = checkpoint

        # Creation of additional attributes for training
        self.sess = None
        self.writer = None
        self.ckp_path, self.summaries_path = None, None
        self.streaming_accuracy_update, self.streaming_accuracy_scalar = None, None
        self.train_step = None
        self.iteration_variable, self.mega_batch_variable = None, None
        self.time_variable, self.it_from_start_variable = None, None
        self.iteration, self.mega_batch, self.it_from_start, self.time, self.aux_tensor = None, None, None, None, None
        self.saver = None
        self.test_iterator = None

    def __prepare(self):
        """
        It does the preparation for the training. This preparations include:
        -Creating a session (if a default session is already in use, then this method deletes it and replaces with a new
        one)
        -Creates operators needed for summaries for TensorBoard
        -Sets the Optimizer
        -Sets variables and operators needed for saving checkpoints of the training
        -Initializes all the variables
        :return: None
        """
        print("Preparing training...")

        self.ckp_path, self.summaries_path = utils.prepare_directories(self.config)

        # Creates the session
        sess = tf.get_default_session()
        if sess:
            sess.close()
        self.sess = tf.InteractiveSession()

        # Creates loss function and summaries
        self.loss = self._create_loss(self.tensor_y, self.model.get_output())
        correct_prediction = tf.equal(tf.argmax(self.tensor_y, 1), tf.argmax(self.model.get_output(), 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.streaming_accuracy, self.streaming_accuracy_update = tf.metrics.mean(accuracy)
        self.streaming_accuracy_scalar = tf.summary.scalar('accuracy', self.streaming_accuracy)

        self.train_step = self._create_optimizer(self.config, self.loss)

        self._prepare_variables_for_checkpoints()

        self.saver = tf.train.Saver()

        self._custom_prepare(self.sess)

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        print("Finished preparations for training...")

    def train(self):
        """
        Trains a neural network with the appropriate configuration. It also does the preparations needed for that
        training. It loads a checkpoint if a valid checkpoint path has been given.
        :return: None
        """
        self.__prepare()
        inc, skip_count, iteration, start_time = self._maybe_load_model(self.checkpoint)

        self.test_iterator, test_x, test_y = self.pipeline.build_test_data_tensor()

        for i in range(inc, len(self.config.train_configurations)):
            self.writer = tf.summary.FileWriter(os.path.join(self.summaries_path, "increment_{}".format(i)),
                                                tf.get_default_graph())
            self.pipeline.change_dataset_part(i)
            training_iterator, data_x, data_y = self.pipeline.build_train_data_tensor(skip_count)
            self.sess.run(training_iterator.initializer)
            iteration = self.train_increment(i, skip_count, iteration, start_time,
                                             self.config.train_configurations[i].ttime,
                                             self.config, self.writer, data_x, data_y, test_x, test_y)
            self._post_process_increment()
            # Reestablishes time and skip_count to zero after the first mega-batch (useful when a checkpoint is loaded)
            start_time = 0
            skip_count = 0
            print("Finished training of increment {}...".format(i))
            self.writer.close()

        self.__finish()

    def train_increment(self, increment: int, iteration: int, total_iteration: int, trained_time: float, ttime: float,
                        config: GeneralConfig,
                        writer: tf.summary.FileWriter,
                        data_x: tf.Tensor, data_y: tf.Tensor,
                        test_x: tf.Tensor, test_y: tf.Tensor):
        """
        Trains a neural network over an increment of data, with the given configuration and model, over the data that is
        provided. It also saves summaries for TensorBoard and creates checkpoints according to the given configuration.
        Note: if both maximum number of epochs and maximum time are set, then the training finishes when any of the
        stop criteria is met
        :param increment: the number of the mega-batch
        :param iteration: the current iteration number over the training data. It should be zero if no checkpoint has
        been loaded or if the mega-batch at which the restored checkpoint is differs from the current mega-batch
        :param total_iteration: the current iteration number over the training data, counting from the start of the
        training (that is, from the first batch of mega-batch 0)
        :param trained_time: number of seconds that the network has been trained (counting from the start of the
        mega-batch). It should be zero if no checkpoint has been loaded or if the current mega-batch is different than
        the one that was loaded from a checkpoint
        :param ttime: number of seconds that the model should be trained. If None, then time restrictions are not used
        :param config: the configuration for the whole training
        :param writer: a FileWriter properly configured
        :param data_x: the tensor associated with the training data
        :param data_y: the tensor that has the corresponding labels of the training data
        :param test_x: the tensor associated with the testing/validation data
        :param test_y: the tensor that has the corresponding labels of the testing/validation data
        :return: the current iteration
        """
        print("Starting training of increment {}...".format(increment))
        start_time = time.time()
        i = total_iteration  # Iteration counting from the start of the training
        self.__perform_validation(i, writer, test_x, test_y)  # Performs validation at the beginning

        while True:
            try:
                image_batch, target_batch = self.sess.run([data_x, data_y])

                _, c = self._train_batch(self.sess, image_batch, target_batch, self.tensor_x, self.tensor_y,
                                         self.train_step, self.loss, increment, iteration, total_iteration)
                curr_time = time.time() + trained_time  # If a checkpoint has been loaded, it should adapt the time
                interval = curr_time - start_time

                if i % config.summary_interval == 0 and not i == 0:
                    print("Performing validation at iteration: {}. Loss is: {}. "
                          "Time is: {}".format(i, c, interval))
                    self.__perform_validation(i, writer, test_x, test_y)
                if i % config.check_interval == 0:
                    self._save_model(iteration, i, increment, interval)
                if ttime and interval > ttime:
                    print("Finished increment {} in {} seconds. You can see the results of the training using "
                          "Tensorboard".format(increment, interval))
                    break

            except OutOfRangeError:
                break
            i += 1
            iteration += 1
        self.__perform_validation(i, writer, test_x, test_y)  # Performs validation at the end
        return i

    def __perform_validation(self, iteration: int, writer: tf.summary.FileWriter,
                             test_x: tf.Tensor, test_y: tf.Tensor):
        """
        Performs validation over the test data and register the results in the form of summaries that can be interpreted
        by Tensorboard
        :param iteration: the current iteration number over the training data
        :param writer: a FileWriter properly configured
        :param test_x: the tensor associated with the testing/validation data
        :param test_y: the tensor that has the corresponding labels of the data
        :return: None
        """
        self.sess.run(self.test_iterator.initializer)
        self.sess.run(tf.variables_initializer(tf.get_default_graph().get_collection(tf.GraphKeys.METRIC_VARIABLES)))
        while True:
            try:
                test_images, test_target = self.sess.run([test_x, test_y])
                self.streaming_accuracy = self.sess.run([self.streaming_accuracy_update],
                                                        feed_dict={self.tensor_x: test_images,
                                                                   self.tensor_y: test_target})
            except OutOfRangeError:
                print("Finished validation of iteration {}...".format(iteration))
                break

        summary = self.sess.run(self.streaming_accuracy_scalar)
        writer.add_summary(summary, iteration)

    def _maybe_load_model(self, ckp_path: str):
        """
        This method prepares the previously created neural network with the checkpoint data if a checkpoint is
        provided. It also loads any kind of additional Variable that is need for the training (like Data or Optimizer's
        variables).
        :param ckp_path: the checkpoint path if it's required to start the training from a checkpoint. A data path with
        the following structure is expected: ./checkpoints/dataset_name/config_name/checkpoint_name.ckpt.
        If there is no checkpoint to be loaded then its value should be None.
        :return:  if a checkpoint has been successfully loaded then this method returns a tuple containing 4 values:
        the number of the current mega-batch (increment), iteration over the batch, iteration counting from the start
        of the training, and the time that the network has already been trained (counting from the start of the
        mega-batch) in that order. It returns a tuple of zeros if no checkpoint is loaded.
        """
        if not ckp_path:
            print("No checkpoint has been loaded.")
            return 0, 0, 0, 0
        else:
            print("Loading checkpoint from {}.".format(ckp_path))

        self.saver.restore(self.sess, ckp_path)
        inc, it, it_t, t = self.sess.run([self.mega_batch_variable, self.iteration_variable,
                                          self.it_from_start_variable, self.time_variable])
        self._custom_checkpoint_load(self.sess)
        print("Loaded checkpoint at iteration {} of increment {}. Total iterations: {}".format(it, inc, it_t))
        return int(inc[0]), int(it[0] + 1), int(it_t[0] + 1), t[0]

    def _save_model(self, iteration: int, total_iteration: int, increment: int, curr_time: float):
        """
        Saves all the variables of the model
        :param iteration: the current iteration number over the training data
        :param increment: the number of the mega-batch
        :param curr_time: the time that has passed since the beginning of the training of the current batch. This time
        must be in seconds
        """
        filename = "model-{}-{}.ckpt".format(increment, total_iteration)
        self.sess.run(self.mega_batch, feed_dict={self.aux_tensor: [increment]})
        self.sess.run(self.iteration, feed_dict={self.aux_tensor: [iteration]})
        self.sess.run(self.it_from_start, feed_dict={self.aux_tensor: [total_iteration]})
        self.sess.run(self.time, feed_dict={self.aux_tensor: [curr_time]})
        self._custom_checkpoint_save(self.sess)
        save_path = self.saver.save(self.sess, os.path.join(self.ckp_path, filename))
        print("Model saved in path: {}".format(save_path))
        return save_path

    def __finish(self):
        """
        Performs the required operations to finish the training (like closing pipelines)
        :return: None
        """
        print("Finishing training...")
        self.pipeline.close()
        self.writer.close()

    def _prepare_variables_for_checkpoints(self):
        """
        Prepares all tensors and variables needed for a proper checkpoint save and load. This method only prepares
        the variables used for the basic version of checkpoint loading
        :return: None
        """
        self.iteration_variable = tf.get_variable("iteration", shape=[1], initializer=tf.zeros_initializer)
        self.mega_batch_variable = tf.get_variable("megabatch", shape=[1], initializer=tf.zeros_initializer)
        self.it_from_start_variable = tf.get_variable("it_start", shape=[1], initializer=tf.zeros_initializer)
        self.time_variable = tf.get_variable("time", shape=[1], initializer=tf.zeros_initializer)
        self.aux_tensor = tf.placeholder(dtype=tf.float32, shape=[None])
        self.iteration = self.iteration_variable.assign(self.aux_tensor)
        self.mega_batch = self.mega_batch_variable.assign(self.aux_tensor)
        self.it_from_start = self.it_from_start_variable.assign(self.aux_tensor)
        self.time = self.time_variable.assign(self.aux_tensor)

    @abstractmethod
    def _create_loss(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        """
        Creates a loss function
        :param tensor_y: the tensor corresponding to the output of a training
        :param net_output: a tensor corresponding to the last layer of a neural network
        :return: a Tensor corresponding to the loss function
        """
        raise NotImplementedError("The subclass hasn't implemented the _create_loss method")

    @abstractmethod
    def _create_optimizer(self, config: GeneralConfig, loss: tf.Tensor):
        """
        Creates the Optimizer for the training (e.g. AdaGradOptimizer)
        :param config: the configuration for the Optimizer
        :param loss: a tensor representing the loss function
        :return: a tf.Optimizer
        """
        raise NotImplementedError("The subclass hasn't implemented the _create_optimizer method")

    def _post_process_increment(self):
        """
        Does some post processing after the training of a batch is completed. It isn't implemented in the base version,
        but may be overridden by a subclass that needs to perform changes to a variable or any kind of process after
        the training.
        :return: None
        """
        pass

    @abstractmethod
    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, loss: tf.Tensor, increment: int, iteration: int, total_it: int):
        """
        Trains the current model over a batch of data
        :param sess: the current Session
        :param image_batch: the batch of data corresponding to the input, as obtained from the data pipeline
        :param target_batch: the batch of data corresponding to the output, as obtained from the data pipeline
        :param tensor_x: the tensor corresponding to the input of a training
        :param tensor_y: the tensor corresponding to the output of a training
        :param train_step: the current tf.Optimizer
        :param loss: a tensor representing the loss function
        :param increment: the number of the mega-batch
        :param iteration: the current iteration counting from the start of the mega-batch
        _param total_it: the current iteration counting from the start of the whole training
        :return: a tuple containing the result of the training, i.e. the result of running train_step and loss (in that
        order) over the batch of data using sess as Session
        """
        raise NotImplementedError("The subclass hasn't implemented the _train_batch method")

    def _custom_prepare(self, sess):
        """
        This method may be used by concrete trainers to define custom preparations for the training. This method isn't
        implemented by default
        :param sess: the current session
        :return: None
        """
        pass

    def _custom_checkpoint_load(self, sess):
        """
        This method may be used by concrete trainers to define custom attributes to be obtained when a checkpoint is
        loaded. This is intended to be used for information that isn't supported in the base Trainer's checkpoint load.
        Please note that the checkpoints are saved and restored using the Saver class from Tensorflow, so all the
        information must be loaded from that source. If you need another kind of checkpoint management then you should
        override the _maybe_load_model and _save_model methods.
        This method isn't implemented by default
        :param sess: the current session
        :return: None
        """
        pass

    def _custom_checkpoint_save(self, sess):
        """
        This method may be used by concrete trainers to define custom attributes to be stored when a checkpoint is
        saved. This is intended to be used for information that isn't supported in the base Trainer's checkpoint saver.
        Please note that the checkpoints are saved and restored using the Saver class from Tensorflow, so all the
        information must be loaded from that source. If you need another kind of checkpoint management then you should
        override the _maybe_load_model and _save_model methods.
        This method isn't implemented by default
        :param sess: the current session
        :return: None
        """
        pass
