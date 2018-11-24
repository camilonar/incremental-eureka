"""
Experiment for MNIST dataset using the training algorithm that uses artificial sampling with DCGAN
"""
from experiments.mnist_exp import MnistExperiment
from training.config.dcgan_config import DCGANConfig
from training.support.tester import Tester
from training.trainer.dcgan_trainer import DCGANTrainer
from training.config.megabatch_config import MegabatchConfig


class MnistExperimentDCGAN(MnistExperiment):
    """
    Performs experiments over MNIST dataset using the training algorithm that uses artificial sampling with DCGAN
    """

    def _prepare_trainer(self):
        tester = Tester(self.neural_net, self.data_input, self.input_tensor, self.output_tensor)
        self.__trainer = DCGANTrainer(self.general_config, self.neural_net, self.data_input, self.input_tensor,
                                      self.output_tensor, tester=tester, checkpoint=self.ckp_path)

    def _prepare_config(self, str_optimizer: str, is_incremental: bool):
        self.__general_config = DCGANConfig(0.0001, dcgan_lr=0.0002, beta1=0.5, input_height=32, input_width=32,
                                            output_height=32, output_width=32, c_dim=1, z_dim=100,
                                            summary_interval=self.summary_interval, check_interval=self.ckp_interval,
                                            config_name=str_optimizer, model_name=self.dataset_name)
        # Creates configuration for 5 mega-batches
        if is_incremental:
            for i in range(5):
                train_conf = MegabatchConfig(50, batch_size=64)
                self.general_config.add_train_conf(train_conf)
        else:
            train_conf = MegabatchConfig(50, batch_size=64)
            self.general_config.add_train_conf(train_conf)

    @property
    def general_config(self):
        return self.__general_config

    @property
    def trainer(self):
        return self.__trainer
