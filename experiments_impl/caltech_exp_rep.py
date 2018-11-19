"""
Experiment for Caltech-101 dataset using the proposed representative-selection algorithm
"""
from experiments.caltech_exp import CaltechExperiment
from training.support.tester import Tester
from training.trainer.rep_trainer import RepresentativesTrainer
from training.config.general_config import GeneralConfig
from training.config.increment_config import IncrementConfig


class CaltechExperimentRep(CaltechExperiment):
    """
    Performs experiments over Caltech-101 dataset using the proposed representative-selection algorithm
    """

    def _prepare_trainer(self):
        tester = Tester(self.neural_net, self.data_input, self.input_tensor, self.output_tensor)
        self.__trainer = RepresentativesTrainer(self.general_config, self.neural_net, self.data_input,
                                                self.input_tensor, self.output_tensor,
                                                tester=tester, checkpoint=self.ckp_path)

    def _prepare_config(self, str_optimizer: str, is_incremental: bool):
        self.__general_config = GeneralConfig(0.0001, self.summary_interval, self.ckp_interval,
                                              config_name=str_optimizer, model_name=self.dataset_name)
        # Creates configuration for 5 mega-batches
        if is_incremental:
            for i in range(5):
                train_conf = IncrementConfig(1, batch_size=160)
                self.general_config.add_train_conf(train_conf)
        else:
            train_conf = IncrementConfig(1, batch_size=160)
            self.general_config.add_train_conf(train_conf)

    @property
    def general_config(self):
        return self.__general_config

    @property
    def trainer(self):
        return self.__trainer