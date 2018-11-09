"""
Experiment for MNIST dataset using the proposed representative-selection algorithm
"""
from experiments.mnist_exp import MnistExperiment
from training.trainer.rep_trainer import RepresentativesTrainer
from training.config.general_config import GeneralConfig
from training.config.increment_config import IncrementConfig


class MnistRepExperiment(MnistExperiment):
    """
    Performs experiments over MNIST dataset using the proposed representative-selection algorithm
    """

    def _prepare_trainer(self):
        self.trainer = RepresentativesTrainer(self.general_config, self.neural_net, self.data_input, self.input_tensor,
                                              self.output_tensor, self.ckp_path)

    def _prepare_config(self, str_optimizer: str, is_incremental: bool):
        self.__general_config = GeneralConfig(0.0001, self.summary_interval, self.ckp_interval,
                                              config_name=str_optimizer, model_name=self.dataset_name)
        # Creates configuration for 5 mega-batches
        if is_incremental:
            for i in range(5):
                train_conf = IncrementConfig(50, batch_size=128)
                self.general_config.add_train_conf(train_conf)
        else:
            train_conf = IncrementConfig(50, batch_size=128)
            self.general_config.add_train_conf(train_conf)

    @property
    def general_config(self):
        return self.__general_config
