"""
Experiment for Tiny Imagenet dataset using the proposed representative-selection algorithm NIL
"""
import copy

from errors import OptionNotSupportedError
from experiments.imagenet.imagenet_exp import ImagenetExperiment
from experiments.tester import Tester
from training.config.crif_config import CRIFConfig
from training.config.megabatch_config import MegabatchConfig
from training.trainer.nil_trainer import NILTrainer
from utils.train_modes import TrainMode
from utils import constants as const


class ImagenetExperimentNIL(ImagenetExperiment):
    """
    Performs experiments over Tiny Imagenet dataset using the proposed representative-selection algorithm NIL
    """
    optimizer_name = const.TR_NIL
    general_config = None
    trainer = None

    def _prepare_trainer(self):
        tester = Tester(self.neural_net, self.data_input, self.input_tensor, self.output_tensor)
        self.trainer = NILTrainer(self.general_config, self.neural_net, self.data_input,
                                  self.input_tensor, self.output_tensor,
                                  tester=tester, checkpoint=self.ckp_path)

    def _prepare_config(self, str_optimizer: str, train_mode: TrainMode):
        self.general_config = CRIFConfig(train_mode, 0.0001, self.summary_interval, self.ckp_interval,
                                         config_name=str_optimizer, model_name=self.dataset_name)
        # Creates configuration for 5 mega-batches
        if train_mode == TrainMode.INCREMENTAL or train_mode == TrainMode.ACUMULATIVE:
            for i in range(5):
                train_conf = MegabatchConfig(1, batch_size=128)
                self.general_config.add_train_conf(train_conf)
        else:
            raise OptionNotSupportedError("The requested Experiment class: {} doesn't support the requested training"
                                          " mode: {}".format(self.__class__, train_mode))

    def _prepare_scenarios(self, base_config):
        scenarios = None
        scenarios = self._add_scenario(scenarios, base_config, 'Test with 1% of data stored as representatives')
        scenario = copy.copy(base_config)
        scenario.memory_size = 500
        scenarios = self._add_scenario(scenarios, scenario, 'Test with 10% of data stored as representatives')
        return scenarios
