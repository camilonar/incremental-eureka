"""
Module that helps with the execution of tests.
"""
from abc import ABC, abstractmethod
from errors import OptimizerNotSupportedError, TestNotPreparedError


class Tester(ABC):
    """
    This class helps with the configuration of the pre-established tests.
    """

    def __init__(self, lr: float, summary_interval=100, check_interval=200):
        """
        It creates a Tester object
        :param lr: the learning rate to be used in the training
        :param summary_interval: the interval of iterations at which the summaries are going to be performed
        :param check_interval: the interval of iterations at which the evaluations and checkpoints are going to be
        performed. Must be an integer multiple of summary_interval
        """
        self.lr = lr
        self.summary_interval = summary_interval
        self.check_interval = check_interval
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
        :param str_optimizer: a string that represents the chosen Optimizer. Currently supported strings are.
            -BASE: for a simple RMSProp
            -CEAL: for the CEAL algorithm (See: Keze Wang, Dongyu Zhang, Ya Li, Ruimao Zhang, and Liang Lin.
                    Cost-effective active learning for deep image classification.
                    IEEE Transactions on Circuits and Systems for Video Technology, 2016)
            -REPRESENTATIVES: for the proposed approach of this work, i.e. an incremental algorithm that uses RMSProp
                    and select samples based in clustering
        :return: None
        :raises OptimizerNotSupportedError: if the required Optimizer isn't supported yet
        """
        if str_optimizer == 'BASE':
            pass  # Base Optimizer (basic RMSProp)
        elif str_optimizer == 'CEAL':
            pass  # CEAL Optimizer
        elif str_optimizer == 'REPRESENTATIVES':
            pass  # Our Optimizer
        else:
            raise OptimizerNotSupportedError("The required Optimizer '{}' isn't supported".format(str_optimizer))

    @abstractmethod
    def _prepare_config(self):
        """
        This method creates and saves the proper Configuration for the training according to the pre-established
        conditions of each dataset
        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the _prepare_config method")

    @abstractmethod
    def _prepare_checkpoint_if_required(self, ckp_path: str):
        """
        This method should prepare the previously created neural network with the checkpoint data if a checkpoint is
        provided, otherwise, this method shouldn't do any kind of modification over the network data. It should also
        check if the checkpoint path is a VALID path.
        :param ckp_path: the checkpoint path if it's required to start the training from a checkpoint. A data path with
        the following structure is expected: ./checkpoints/dataset_name/neural_net_name/checkpoint_name.ckpt.
        If there is no checkpoint to be loaded then its value should be None.
        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the _prepare_checkpoint_if_required method")

    def prepare_all(self, str_optimizer: str, ckp_path: str = None):
        """
        It prepares the Tester object for the test, according to the various parameters given up to this point and
        also according to the corresponding dataset to which the concrete Tester is associated.
        :param str_optimizer: str_optimizer: a string that represents the chosen Optimizer. Currently supported strings are.
            -BASE: for a simple RMSProp
            -CEAL: for the CEAL algorithm (See: Keze Wang, Dongyu Zhang, Ya Li, Ruimao Zhang, and Liang Lin.
                    Cost-effective active learning for deep image classification.
                    IEEE Transactions on Circuits and Systems for Video Technology, 2016)
            -REPRESENTATIVES: for the proposed approach of this work, i.e. an incremental algorithm that uses RMSProp
                    and select samples based in clustering
        :param ckp_path: the checkpoint path if it's required to start the training from a checkpoint. A data path with
        the following structure is expected: ./checkpoints/dataset_name/neural_net_name/checkpoint_name.ckpt.
        If there is no checkpoint to be loaded then its value should be None. The default value is None.
        :return:
        """
        self._prepare_data_pipeline()
        self._prepare_neural_network()
        self._prepare_optimizer(str_optimizer)
        self._prepare_config()
        self._prepare_checkpoint_if_required(ckp_path)

    # TODO implementar execute_test
    def execute_test(self):
        """
        Calls the trainer to perform the test with the given configuration. It should raise an exception if the _prepare
        methods (or prepare_all) hasn't been executed before this method.
        :return: None
        :raises TestNotPreparedError: if the Tester hasn't been prepared before the execution of this method
        """
        self.__check_conditions_for_test()

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
        if not self.data_input_loaded:
            message += '-Data pipeline missing\n'
        if not self.neural_net_loaded:
            message += '-Neural Network missing\n'
        if not self.optimizer:
            message += '-Optimizer missing\n'
        if not self.train_config_loaded:
            message += '-Training Configuration missing\n'
        if not self.checkpoint_loaded:
            message += '-Checkpoint required by user, but not loaded'

        if message:
            raise TestNotPreparedError("There has been some problems when checking the requirements for the execution"
                                       " of the test:\n {}".format(message))

    @property
    @abstractmethod
    def data_input_loaded(self):
        """
        It tells whether or not the Data input object has been properly created and established into the Tester
        :return: True if the Data input object has been properly created, and False in case that it hasn't been created
        at all
        """
        pass

    @property
    @abstractmethod
    def neural_net_loaded(self):
        """
        It tells whether or not the neural net object has been properly created and established into the Tester
        :return: True if the Network object has been properly created, and False in case that it hasn't been created
        at all
        """
        pass

    @property
    @abstractmethod
    def train_config_loaded(self):
        """
        It tells whether or not the training configuration object has been properly created and established into the
        Tester
        :return: True if the Trainer object has been properly created, and False in case that it hasn't been created
        at all
        """
        pass

    @property
    @abstractmethod
    def checkpoint_loaded(self):
        """
        It tells whether or not a checkpoint for the training has been loaded, in case that a checkpoint has been
        required by the User
        :return: it should return True if the checkpoint has been properly loaded into the neural net or if no
        checkpoint has been requested. It should return false if a checkpoint has been requested but hasn't been loaded
        into the net
        """
        pass
