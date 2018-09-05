"""
Module for the configuration of training that is going to be used.
There are two classes:
    1. GeneralConfig, which has the configuration data for the whole training (including the incremental learning)
    2. TrainConfig, which has the configuration data for a specific increment of data
"""


class TrainConfig(object):
    """
    Configuration for training based on a batch of data. This is used for a specific increment of data
    """

    def __init__(self, epochs: int, ttime: int = None, batch_size=100):
        """
        Creates a TrainConfig object
        :param epochs: number of epochs for the training. If None, then the dataset is repeated forever
        :param ttime: number of seconds that the model should be trained. If None, then time restrictions are not used
        :param batch_size: the sizes of the batches that are going to be used in training. This is different from the
        number of instances of each incremental training. E.g. An incremental training may be of a total of 1000
        samples using batches with batch_size 100
        """
        self.epochs = epochs
        self.ttime = ttime
        self.batch_size = batch_size


class GeneralConfig(object):
    """
    General configuration that it's used for the whole training. E.g. if the training is composed of 5 different batches
    of data, then this part of the configuration is used for ALL of them
    """

    def __init__(self, learning_rate: float,
                 summary_interval=100, check_interval=200, config_name='default', model_name='dataset_default'):
        """
        Creates a GeneralConfig object
        :param learning_rate: the learning rate to be used in the training
        :param summary_interval: the interval of iterations at which the summaries are going to be performed
        :param check_interval: the interval of iterations at which the evaluations and checkpoints are going to be
        performed. Must be an integer multiple of summary_interval
        :param config_name: a descriptive name for the training configuration
        :param model_name: a descriptive name for the model
        """
        self.learn_rate = learning_rate
        self.summary_interval = summary_interval
        self.check_interval = check_interval
        self.config_name = config_name
        self.model_name = model_name
        self.train_configurations = []  # It stores the configurations for each mega batch of training data
        # model_name = 'vgg_16'  # choose model
        # model = staticmethod(globals()[model_name])  # gets model by name

    def add_train_conf(self, train_conf: TrainConfig):
        """
        Adds a TrainConfig object to the list of mega-batch-specific configurations. The objects must be added in
        the same order as how they are going to be used (i.e. the configuration for the first batch must be added first,
        for the second batch must be added second, ...)
        :param train_conf: a configuration for a mega batch of data
        :return: None
        """
        self.train_configurations.append(train_conf)
