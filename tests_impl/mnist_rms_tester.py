"""
Tester for MNIST dataset using base RMSProp
"""
from tests.mnist_tester import MnistTester
from training.basic_trainer import RMSPropTrainer
from training.train_conf import GeneralConfig, TrainConfig


class MnistRMSPropTester(MnistTester):
    """
    Performs tests over MNIST dataset using RMSProp
    """

    def _prepare_trainer(self):
        self.trainer = RMSPropTrainer(self.general_config, self.neural_net, self.data_input, self.input_tensor,
                                      self.output_tensor, self.ckp_path)

    def _prepare_config(self, str_optimizer: str):
        self.__general_config = GeneralConfig(0.01, self.summary_interval, self.ckp_interval,
                                              config_name=str_optimizer, model_name=self.dataset_name)
        # Creates configuration for 5 mega-batches
        for i in range(5):
            train_conf = TrainConfig(2, batch_size=128)
            self.general_config.add_train_conf(train_conf)

    @property
    def general_config(self):
        return self.__general_config