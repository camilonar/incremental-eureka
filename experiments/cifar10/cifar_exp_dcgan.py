"""
Experiment for Cifar-10 dataset using the training algorithm that uses artificial sampling with DCGAN
"""
from errors import OptionNotSupportedError
from experiments.cifar10.cifar_exp import CifarExperiment
from training.support.tester import Tester
from training.trainer.dcgan_trainer import DCGANTrainer
from training.config.general_config import GeneralConfig
from training.config.megabatch_config import MegabatchConfig
from utils.train_modes import TrainMode


class CifarExperimentDCGAN(CifarExperiment):
    """
    Performs experiments over Cifar-10 dataset using the training algorithm that uses artificial sampling with DCGAN
    """
    general_config = None
    trainer = None

    def _prepare_trainer(self):
        tester = Tester(self.neural_net, self.data_input, self.input_tensor, self.output_tensor)
        self.trainer = DCGANTrainer(self.general_config, self.neural_net, self.data_input, self.input_tensor,
                                    self.output_tensor, tester=tester, checkpoint=self.ckp_path)

    def _prepare_config(self, str_optimizer: str, train_mode: TrainMode):
        self.general_config = GeneralConfig(train_mode, 0.0001, self.summary_interval, self.ckp_interval,
                                            config_name=str_optimizer, model_name=self.dataset_name)
        # Creates configuration for 5 mega-batches
        if train_mode == TrainMode.INCREMENTAL or train_mode == TrainMode.ACUMULATIVE:
            for i in range(5):
                train_conf = MegabatchConfig(300, batch_size=64)
                self.general_config.add_train_conf(train_conf)
        else:
            raise OptionNotSupportedError("The requested Experiment class: {} doesn't support the requested training"
                                          " mode: {}".format(self.__class__, train_mode))
