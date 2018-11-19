"""
Module that helps with the execution of experiments.
Features:
1. Prepares the ambient for testing
2. Executes a test training a neural network over a dataset according to a flexible configuration
"""
from abc import ABC, abstractmethod
from errors import ExperimentNotPreparedError

import utils.dir_utils as dir
import tensorflow as tf
import numpy as np
import utils.constants as const


class Experiment(ABC):
    """
    This class helps with the configuration of the pre-established experiments.
    """

    def __init__(self, train_dirs: [str], validation_dir: str,
                 summary_interval=100, ckp_interval=200, checkpoint_key: str = None):
        """
        It creates an Experiment object
        :param train_dirs: array of strings corresponding to the paths of each one of the mega-batches for training
        :param validation_dir: a string corresponding to the path of the testing data
        :param summary_interval: the interval of iterations at which the summaries are going to be performed
        :param ckp_interval: the interval of iterations at which the evaluations and checkpoints are going to be
        performed. Must be an integer multiple of summary_interval
        :param checkpoint_key: a string containing the checkpoint's corresponding mega-batch and iteration if it's
        required to start the training from a checkpoint. It is expected to follow the format
        "[mega-batch]-[iteration]", e.g. "0-50".
        If there is no checkpoint to be loaded then its value should be None. The default value is None.

        This must be called by the constructors of the subclasses.
        """
        self.train_dirs = train_dirs
        self.validation_dir = validation_dir
        self.summary_interval = summary_interval
        self.ckp_interval = ckp_interval
        self.checkpoint_key = checkpoint_key
        self.ckp_path = None

    @abstractmethod
    def _prepare_data_pipeline(self):
        """
        It prepares the data pipeline according to the configuration of each Experiment
        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the _prepare_data_pipeline method")

    @abstractmethod
    def _prepare_neural_network(self):
        """
        It creates and stores the proper neural network according to the assigned dataset of the tester.
        E.g. if the Experiment performs experiments over ImageNet then it should create a CaffeNet, but if the experiments are over
        MNIST then it should create a LeNet.
        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the _prepare_neural_network method")

    @abstractmethod
    def _prepare_trainer(self):
        """
        Prepares the trainer that is required by the User. All the other preparations (e.g. _prepare_config,
        _prepare_neural_network) must be completed before this method is used, otherwise, there may be unexpected
        behavior
        :return: None
        """

    @abstractmethod
    def _prepare_config(self, str_optimizer: str, is_incremental: bool):
        """
        This method creates and saves the proper Configuration for the training according to the pre-established
        conditions of each dataset
        :param str_optimizer: a string that represents the chosen Trainer.
        :param is_incremental: True to indicate that the training is gonna contain multiple mega-batches
        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the _prepare_config method")

    def _prepare_checkpoint_if_required(self, checkpoint_key: str, str_optimizer: str):
        """
        This method prepares the checkpoint path given an incomplete checkpoint path. It also checks if the created
        checkpoint path is a valid path.
        :param checkpoint_key: the checkpoint key if it's required to start the training from a checkpoint. It is
         expected to follow the format "[increment]-[iteration]", e.g. "0-50".
        If there is no checkpoint to be loaded then its value should be None.
        :param str_optimizer: a string that represents the selected trainer/optimizer
        :return: if a checkpoint has been successfully loaded then this method returns a string representing the full
        path to the checkpoint. If no checkpoint has been requested or if the generated path doesn't exists
        then this method returns None
        """
        if checkpoint_key:
            path, valid = dir.create_full_checkpoint_path(self.dataset_name, str_optimizer, checkpoint_key)
            if valid:
                print("The checkpoint will be loaded from: {}".format(path))
                return path
        print("No checkpoint has been loaded...")
        return None

    def prepare_all(self, str_trainer: str, is_incremental: bool):
        """
        It prepares the Experiment object for the test, according to the various parameters given up to this point and
        also according to the corresponding dataset to which the concrete Experiment is associated.
        This method calls ALL the _prepare methods defined in the base class.
        :param str_trainer: a string that represents the chosen Trainer. Currently supported strings are:
            -OPT_BASE: for a simple RMSProp
            -OPT_DCGAN: for the Trainer that uses the algorithm presented in "Evolutive deep models for online learning
                    on data streams with no storage"
                    See: http://ceur-ws.org/Vol-1958/IOTSTREAMING2.pdf
            -OPT_REPRESENTATIVES: for the proposed approach of this work, i.e. an incremental algorithm that uses RMSProp
                    and select samples based in clustering
        :param is_incremental: True to indicate that the training is gonna contain multiple mega-batches
        :return: None
        """
        tf.reset_default_graph()
        tf.set_random_seed(const.SEED)
        np.random.seed(const.SEED)
        self._prepare_config(str_trainer, is_incremental)
        self._prepare_data_pipeline()
        self._prepare_neural_network()
        self.ckp_path = self._prepare_checkpoint_if_required(self.checkpoint_key, str_trainer)
        self._prepare_trainer()

    def execute_experiment(self):
        """
        Calls the trainer to perform the experiment with the given configuration. It should raise an exception if the
        _prepare methods (or prepare_all) hasn't been executed before this method.
        :return: None
        :raises ExperimentNotPreparedError: if the Experiment hasn't been prepared before the execution of this method
        """
        self.__check_conditions_for_experiment()
        self.trainer.train()

    def __check_conditions_for_experiment(self):
        """
        Checks if the Experiment is ready to perform a test. The evaluated requirements are:
        -The data pipeline
        -The Neural Network
        -The Optimizer
        -The training configuration
        -If a checkpoint has been required, then it must have been loaded
        :return: None
        :raises ExperimentNotPreparedError: if it is found that at least one of the prerequisites for the test hasn't been
        fulfilled
        """
        print("Checking conditions for test...")
        message = ""
        if not self.data_input:
            message += '-Data pipeline missing\n'
        if not self.neural_net:
            message += '-Neural Network missing\n'
        if not self.trainer:
            message += '-Trainer algorithm missing\n'
        if not self.general_config:
            message += '-Training Configuration missing\n'
        if not self.checkpoint_loaded:
            message += '-Checkpoint required by user, but not loaded'

        if message:
            raise ExperimentNotPreparedError(
                "There has been some problems when checking the requirements for the execution"
                " of the test:\n {}".format(message))

        print("The test has been properly prepared...")

    @property
    @abstractmethod
    def dataset_name(self):
        """
        Getter for the name of the dataset associated with the Experiment
        :return: the name of the dataset of the Experiment
        """
        pass

    @property
    @abstractmethod
    def data_input(self):
        """
        Getter for the Data pipeline object
        :return: the data pipeline object of the Experiment
        """
        pass

    @property
    @abstractmethod
    def neural_net(self):
        """
        Getter for the Neural network object
        :return: the Neural network object of the Experiment
        """
        pass

    @property
    @abstractmethod
    def general_config(self):
        """
        Getter for the GeneralTraining object
        :return: the GeneralTraining object of the Experiments
        """
        pass

    @property
    @abstractmethod
    def trainer(self):
        """
        Getter for the Trainer object
        :return: the Trainer object of the Experiment
        """
        pass

    @property
    def checkpoint_loaded(self):
        """
        It tells whether or not a checkpoint for the training has been loaded, in case that a checkpoint has been
        required by the User
        :return: it should return True if the checkpoint has been properly loaded into the neural net or if no
        checkpoint has been requested. It should return false if a checkpoint has been requested but hasn't been loaded
        into the net
        """
        if self.checkpoint_key is None:
            return True
        return self.ckp_path

    @property
    @abstractmethod
    def input_tensor(self):
        """
        Getter for the input tensor of the neural network used by the Experiment
        :return: a Tensor that was assigned as 'data' when the network was created
        """
        pass

    @property
    @abstractmethod
    def output_tensor(self):
        """
        Getter for the output tensor of the training
        :return: the Tensor associated to the labels of supervised learning.
        E.g. if the tensor is used for calculating the mse, as follows, then 'data_y' should be the returned Tensor
        of this function:
            mse = tf.reduce_mean(tf.square(data_y - neural_net.get_output()))
        """
        pass