"""
Experiment for MNIST dataset using base RMSProp
"""
from errors import OptionNotSupportedError
from experiments.fashion_mnist.fashion_mnist_exp import FashionMnistExperiment
from training.support.tester import Tester
from training.trainer.rms_trainer import RMSPropTrainer
from training.config.general_config import GeneralConfig
from training.config.megabatch_config import MegabatchConfig
from utils.train_modes import TrainMode


class FashionMnistExperimentRMSProp(FashionMnistExperiment):
    """
    Performs experiments over FASHION-MNIST dataset using RMSProp
    """
    general_config = None
    trainer = None

    def _prepare_trainer(self):
        tester = Tester(self.neural_net, self.data_input, self.input_tensor, self.output_tensor)
        self.trainer = RMSPropTrainer(self.general_config, self.neural_net, self.data_input,
                                      self.input_tensor, self.output_tensor, tester=tester, checkpoint=self.ckp_path)

    def _prepare_config(self, str_optimizer: str, train_mode: TrainMode):
        self.general_config = GeneralConfig(train_mode, 0.0001, self.summary_interval, self.ckp_interval,
                                            config_name=str_optimizer, model_name=self.dataset_name)
        # Creates configuration for 5 mega-batches
        if train_mode == TrainMode.INCREMENTAL or train_mode == TrainMode.ACUMULATIVE:
            for i in range(5):
                train_conf = MegabatchConfig(100, batch_size=256)
                self.general_config.add_train_conf(train_conf)
        else:
            raise OptionNotSupportedError("The requested Experiment class: {} doesn't support the requested training"
                                          " mode: {}".format(self.__class__, train_mode))
