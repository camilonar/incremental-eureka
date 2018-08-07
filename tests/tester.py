"""
Module that helps with the execution of tests.
"""
from errors import OptimizerNotSupportedError


class Tester(object):
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

    def _prepare_data_pipeline(self):
        """
        It prepares the data pipeline according to the configuration of each Tester
        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the _prepare_data_pipeline method")

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

    def _prepare_config(self):
        """
        This method creates and saves the proper Configuration for the training according to the pre-established
        conditions of each dataset
        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the _prepare_config method")

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
    # TODO ver cómo puede saber el método que los _prepare ya han sido ejecutados (chequeando que los atributos no estén vacíos???)
    def execute_test(self):
        """
        Calls the trainer to perform the test with the given configuration. It should raise an exception if the _prepare
        methods (or prepare_all) hasn't been executed before this method.
        :return: None
        :raises TestNotPreparedError: if the Tester hasn't been prepared before the execution of this method
        """
        pass
