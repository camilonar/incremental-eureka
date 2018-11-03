"""
Configuration for the training algorithm that uses artificial sampling with DCGAN
"""
from training.train_conf import GeneralConfig


class DCGANConfig(GeneralConfig):
    """
    Training configuration for the training algorithm that uses artificial sampling with DCGAN
    """

    def __init__(self, learning_rate: float, dcgan_lr: float, beta1: float, input_height: float, input_width: float,
                 output_height: float, output_width: float, c_dim: int, z_dim: int,
                 summary_interval=100, check_interval=200, config_name='default', model_name='dataset_default'):
        """
        Creates a DCGANConfig object
        :param learning_rate: the learning rate to be used in the training
        :param dcgan_lr: the learning rate for Adam, which is used to train the DCGAN networks
        :param beta1: the beta value for Adam, which is used to train the DCGAN networks
        :param input_height: height of input images
        :param input_width: width of input images
        :param output_height: height of output images of DCGAN
        :param output_width: width of output images of DCGAN
        :param c_dim: number of channels of input images of DCGAN
        :param z_dim: Dimension of dim for Z.
        :param summary_interval: the interval of iterations at which the summaries are going to be performed
        :param check_interval: the interval of iterations at which the evaluations and checkpoints are going to be
        performed. Must be an integer multiple of summary_interval
        :param config_name: a descriptive name for the training configuration
        :param model_name: a descriptive name for the model
        """
        super().__init__(learning_rate, summary_interval, check_interval, config_name, model_name)
        self.dcgan_lr = dcgan_lr
        self.beta1 = beta1
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.c_dim = c_dim
        self.z_dim = z_dim
