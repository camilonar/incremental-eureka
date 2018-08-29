"""
Module that helps with the execution of tests.
Features:
1. Prepares the ambient for testing
2. Executes a test training a neural network over a dataset according to a flexible configuration
"""
from abc import ABC, abstractmethod
from errors import OptionNotSupportedError, TestNotPreparedError
from training.trainer import Trainer
import utils.constants as const
import utils.dir_utils as dir

class Tester(ABC):
    """
    This class helps with the configuration of the pre-established tests.
    """

    def __init__(self, lr: float, train_dirs: [str], validation_dir: str, extras: [str],
                 summary_interval=100, ckp_interval=200, inc_ckp_path: str = None):
        """
        It creates a Tester object
        :param train_dirs: array of strings corresponding to the paths of each one of the mega-batches for training
        :param validation_dir: a string corresponding to the path of the testing data
        :param extras: an array of strings corresponding to paths specific for each dataset. It should be an empty array
        :param lr: the learning rate to be used in the training
        :param summary_interval: the interval of iterations at which the summaries are going to be performed
        :param ckp_interval: the interval of iterations at which the evaluations and checkpoints are going to be
        performed. Must be an integer multiple of summary_interval
        :param inc_ckp_path: a string contsining the checkpoint's corresponding mega-batch and iteration if it's
        required to start the training from a checkpoint. It is expected to follow the format
        "[mega-batch]-[iteration]", e.g. "0-50".
        If there is no checkpoint to be loaded then its value should be None. The default value is None.

        This must be called by the constructors of the subclasses.
        """
        self.lr = lr
        self.train_dirs = train_dirs
        self.validation_dir = validation_dir
        self.extras = extras
        self.summary_interval = summary_interval
        self.ckp_interval = ckp_interval
        self.inc_ckp_path = inc_ckp_path
        self.ckp_path = None
        self.optimizer = None

    @abstractmethod
    def _prepare_data_pipeline(self):
        """
        It prepares the data pipeline according to the configuration of each Tester
        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the _prepare_data_pipeline method")

    @abstractmethod
    def _prepare_neural_network(self):
        """
        It creates and stores the proper neural network according to the assigned dataset of the tester.
        E.g. if the Tester performs tests over ImageNet then it should create a CaffeNet, but if the tests are over
        MNIST then it should create a LeNet.
        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the _prepare_neural_network method")

    # TODO implementar las subclases Optimizer
    def _prepare_optimizer(self, str_optimizer: str):

        """
        Prepares the optimizer that is required by the User
        :param str_optimizer: a string that represents the chosen Optimizer. Currently supported strings are:
            -OPT_BASE: for a simple RMSProp
            -OPT_CEAL: for the OPT_CEAL algorithm (See: Keze Wang, Dongyu Zhang, Ya Li, Ruimao Zhang, and Liang Lin.
                    Cost-effective active learning for deep image classification.
                    IEEE Transactions on Circuits and Systems for Video Technology, 2016)
            -OPT_REPRESENTATIVES: for the proposed approach of this work, i.e. an incremental algorithm that uses RMSProp
                    and select samples based in clustering
        :return: None
        :raises OptimizerNotSupportedError: if the required Optimizer isn't supported yet
        """
        self.optimizer = str_optimizer  # TODO cambiarlo cuando existan los Optimizer

        if str_optimizer == const.OPT_BASE:
            pass  # Base Optimizer (basic RMSProp)
        elif str_optimizer == const.OPT_CEAL:
            pass  # OPT_CEAL Optimizer
        elif str_optimizer == const.OPT_REPRESENTATIVES:
            pass  # Our Optimizer
        else:
            raise OptionNotSupportedError("The required Optimizer '{}' isn't supported".format(str_optimizer))

    @abstractmethod
    def _prepare_config(self, str_optimizer: str):
        """
        This method creates and saves the proper Configuration for the training according to the pre-established
        conditions of each dataset
        :type str_optimizer: a string that represents the chosen Optimizer.
        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the _prepare_config method")

    def _prepare_checkpoint_if_required(self, inc_ckp_path: str):
        """
        This method prepares the checkpoint path given an incomplete checkpoint path. It also checks if the created
        checkpoint path is a valid path.
        :param inc_ckp_path: the checkpoint path if it's required to start the training from a checkpoint. It is
         expected to follow the format "[increment]-[iteration]", e.g. "0-50".
        If there is no checkpoint to be loaded then its value should be None.
        :return: if a checkpoint has been successfully loaded then this method returns a string representing the full
        path to the checkpoint. If no checkpoint has been requested or if the generated path doesn't exists
        then this method returns None
        """
        if inc_ckp_path:
            path, valid = dir.create_full_checkpoint_path(self.dataset_name, self.optimizer, self.inc_ckp_path)
            if valid:
                print("The checkpoint will be loaded from: {}".format(path))
                return path
        print("No checkpoint has been loaded...")
        return None

    def prepare_all(self, str_optimizer: str):
        """
        It prepares the Tester object for the test, according to the various parameters given up to this point and
        also according to the corresponding dataset to which the concrete Tester is associated.
        :param str_optimizer: str_optimizer: a string that represents the chosen Optimizer. Currently supported strings are.
            -OPT_BASE: for a simple RMSProp
            -OPT_CEAL: for the OPT_CEAL algorithm (See: Keze Wang, Dongyu Zhang, Ya Li, Ruimao Zhang, and Liang Lin.
                    Cost-effective active learning for deep image classification.
                    IEEE Transactions on Circuits and Systems for Video Technology, 2016)
            -OPT_REPRESENTATIVES: for the proposed approach of this work, i.e. an incremental algorithm that uses RMSProp
                    and select samples based in clustering
        :return: None
        """
        self._prepare_config(str_optimizer)
        self._prepare_data_pipeline()
        self._prepare_neural_network()
        self._prepare_optimizer(str_optimizer)
        self.ckp_path = self._prepare_checkpoint_if_required(self.inc_ckp_path)

    def execute_test(self):
        """
        Calls the trainer to perform the test with the given configuration. It should raise an exception if the _prepare
        methods (or prepare_all) hasn't been executed before this method.
        :return: None
        :raises TestNotPreparedError: if the Tester hasn't been prepared before the execution of this method
        """
        self.__check_conditions_for_test()
        trainer = Trainer(self.general_config, self.neural_net, self.data_input, self.input_tensor, self.output_tensor,
                          self.ckp_path)
        trainer.train()

    def __check_conditions_for_test(self):
        """
        Checks if the Tester is ready to perform a test. The evaluated requirements are:
        -The data pipeline
        -The Neural Network
        -The Optimizer
        -The training configuration
        -If a checkpoint has been required, then it must have been loaded
        :return: None
        :raises TestNotPreparedError: if it is found that at least one of the prerequisites for the test hasn't been
        fulfilled
        """
        print("Checking conditions for test...")
        message = ""
        if not self.data_input:
            message += '-Data pipeline missing\n'
        if not self.neural_net:
            message += '-Neural Network missing\n'
        if not self.optimizer:
            message += '-Optimizer missing\n'
        if not self.general_config:
            message += '-Training Configuration missing\n'
        if not self.checkpoint_loaded:
            message += '-Checkpoint required by user, but not loaded'

        if message:
            raise TestNotPreparedError("There has been some problems when checking the requirements for the execution"
                                       " of the test:\n {}".format(message))

        print("The test has been properly prepared...")

    @property
    @abstractmethod
    def dataset_name(self):
        """
        Getter for the name of the dataset associated with the Tester
        :return: the name of the dataset of the Tester
        """
        pass

    @property
    @abstractmethod
    def data_input(self):
        """
        Getter for the Data pipeline object
        :return: the data pipeline object of the Tester
        """
        pass

    @property
    @abstractmethod
    def neural_net(self):
        """
        Getter for the Neural network object
        :return: the Neural network object of the Tester
        """
        pass

    @property
    @abstractmethod
    def general_config(self):
        """
        Getter for the GeneralTraining object
        :return: the GeneralTraining object of the Testers
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
        if self.inc_ckp_path is None:
            return True
        return self.ckp_path

    @property
    @abstractmethod
    def input_tensor(self):
        """
        Getter for the input tensor of the neural network used by the Tester
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
