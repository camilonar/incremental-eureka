"""
Experiment for MNIST dataset using base RMSProp
"""
from experiments.mnist_exp import MnistExperiment
from training.support.tester import Tester
from training.trainer.rms_trainer import RMSPropTrainer
from training.config.general_config import GeneralConfig
from training.config.increment_config import IncrementConfig


class MnistExperimentRMSProp(MnistExperiment):
    """
    Performs experiments over MNIST dataset using RMSProp
    """

    def _prepare_trainer(self):
        tester = Tester(self.neural_net, self.data_input, self.input_tensor, self.output_tensor)
        self.__trainer = RMSPropTrainer(self.general_config, self.neural_net, self.data_input,
                                        self.input_tensor, self.output_tensor, tester=tester, checkpoint=self.ckp_path)

    def _prepare_config(self, str_optimizer: str, is_incremental: bool):
        self.__general_config = GeneralConfig(0.0001, self.summary_interval, self.ckp_interval,
                                              config_name=str_optimizer, model_name=self.dataset_name)
        # Creates configuration for 5 mega-batches
        if is_incremental:

            for i in range(5):
                train_conf = IncrementConfig(50, batch_size=128)
                self.general_config.add_train_conf(train_conf)
        else:
            self.__general_config.learn_rate = 0.01
            train_conf = IncrementConfig(50, batch_size=128)
            self.general_config.add_train_conf(train_conf)

    @property
    def general_config(self):
        return self.__general_config

    @property
    def trainer(self):
        return self.__trainer
