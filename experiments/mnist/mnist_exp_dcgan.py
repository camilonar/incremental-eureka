"""
Experiment for MNIST dataset using the training algorithm that uses artificial sampling with DCGAN
"""
from errors import OptionNotSupportedError
from experiments.mnist.mnist_exp import MnistExperiment
from training.config.dcgan_config import DCGANConfig
from experiments.tester import Tester
from training.trainer.dcgan_trainer import DCGANTrainer
from training.config.megabatch_config import MegabatchConfig
from utils.train_modes import TrainMode


class MnistExperimentDCGAN(MnistExperiment):
    """
    Performs experiments over MNIST dataset using the training algorithm that uses artificial sampling with DCGAN
    """
    general_config = None
    trainer = None

    def _prepare_trainer(self):
        tester = Tester(self.neural_net, self.data_input, self.input_tensor, self.output_tensor)
        self.trainer = DCGANTrainer(self.general_config, self.neural_net, self.data_input, self.input_tensor,
                                    self.output_tensor, tester=tester, checkpoint=self.ckp_path)

    def _prepare_config(self, str_optimizer: str, train_mode: TrainMode):
        self.general_config = DCGANConfig(train_mode, 0.0001, dcgan_lr=0.0002, beta1=0.5, input_height=32,
                                          input_width=32,
                                          output_height=32, output_width=32, c_dim=1, z_dim=100,
                                          summary_interval=self.summary_interval, check_interval=self.ckp_interval,
                                          config_name=str_optimizer, model_name=self.dataset_name)
        # Creates configuration for 5 mega-batches
        if train_mode == TrainMode.INCREMENTAL or train_mode == TrainMode.ACUMULATIVE:
            for i in range(5):
                train_conf = MegabatchConfig(50, batch_size=64)
                self.general_config.add_train_conf(train_conf)
        else:
            raise OptionNotSupportedError("The requested Experiment class: {} doesn't support the requested training"
                                          " mode: {}".format(self.__class__, train_mode))
