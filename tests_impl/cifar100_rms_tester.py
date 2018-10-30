"""
Tester for Cifar-10 dataset using base RMSProp
"""
from tests.cifar100_tester import Cifar100Tester
from training.rms_trainer import RMSPropTrainer
from training.train_conf import GeneralConfig, TrainConfig


class Cifar100RMSPropTester(Cifar100Tester):
    """
    Performs tests over Cifar-100 dataset using RMSProp
    """

    def _prepare_trainer(self):
        self.trainer = RMSPropTrainer(self.general_config, self.neural_net, self.data_input, self.input_tensor,
                                      self.output_tensor, self.ckp_path)


    def _prepare_config(self, str_optimizer: str, is_incremental: bool):
        self.__general_config = GeneralConfig(0.001, self.summary_interval, self.ckp_interval,
                                              config_name=str_optimizer, model_name=self.dataset_name)
        # Creates configuration for 5 mega-batches
        if is_incremental:
            for i in range(5):
                train_conf = TrainConfig(100, batch_size=128)
                self.general_config.add_train_conf(train_conf)
        else:
            train_conf = TrainConfig(100, batch_size=128)
            self.general_config.add_train_conf(train_conf)

    @property
    def general_config(self):
        return self.__general_config
