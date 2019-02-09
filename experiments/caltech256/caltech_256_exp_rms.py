"""
Experiment for Caltech-101 dataset using base RMSProp
"""
from experiments.caltech256.caltech_256_exp import Caltech256Experiment
from training.support.tester import Tester
from training.trainer.rms_trainer import RMSPropTrainer
from training.config.general_config import GeneralConfig
from training.config.megabatch_config import MegabatchConfig


class Caltech256ExperimentRMSProp(Caltech256Experiment):
    """
    Performs experiments over Caltech-256 dataset using RMSProp
    """
    general_config = None
    trainer = None

    def _prepare_trainer(self):
        tester = Tester(self.neural_net, self.data_input, self.input_tensor, self.output_tensor)
        self.trainer = RMSPropTrainer(self.general_config, self.neural_net, self.data_input, self.input_tensor,
                                      self.output_tensor, tester=tester, checkpoint=self.ckp_path)

    def _prepare_config(self, str_optimizer: str, is_incremental: bool):
        self.general_config = GeneralConfig(0.001, self.summary_interval, self.ckp_interval,
                                            config_name=str_optimizer, model_name=self.dataset_name)
        # Creates configuration for 5 mega-batches
        if is_incremental:
            for i in range(5):
                train_conf = MegabatchConfig(100, batch_size=128)
                self.general_config.add_train_conf(train_conf)
        else:
            train_conf = MegabatchConfig(100, batch_size=128)
            self.general_config.add_train_conf(train_conf)
