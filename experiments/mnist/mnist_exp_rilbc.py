"""
Experiment for MNIST dataset using the proposed representative-selection algorithm RILBC
"""
from errors import OptionNotSupportedError
from experiments.mnist.mnist_exp import MnistExperiment
from experiments.tester import Tester
from training.config.cril_config import CRILConfig
from training.config.megabatch_config import MegabatchConfig
from training.trainer.rilbc_trainer import RILBCTrainer
from utils.train_modes import TrainMode
from utils import constants as const


class MnistExperimentRILBC(MnistExperiment):
    """
    Performs experiments over MNIST dataset using the proposed representative-selection algorithm RILBC
    """
    optimizer_name = const.TR_RILBC
    general_config = None
    trainer = None

    def _prepare_trainer(self):
        tester = Tester(self.neural_net, self.data_input, self.input_tensor, self.output_tensor)
        self.trainer = RILBCTrainer(self.general_config, self.neural_net, self.data_input,
                                    self.input_tensor, self.output_tensor,
                                    tester=tester, checkpoint=self.ckp_path)

    def _prepare_config(self, str_optimizer: str, train_mode: TrainMode):
        self.general_config = CRILConfig(train_mode, 0.0001, self.summary_interval, self.ckp_interval,
                                         config_name=str_optimizer, model_name=self.dataset_name,
                                         n_candidates=50, buffer_size=1)
        # Creates configuration for 5 mega-batches
        if train_mode == TrainMode.INCREMENTAL or train_mode == TrainMode.ACUMULATIVE:
            for i in range(5):
                train_conf = MegabatchConfig(25, batch_size=128)
                self.general_config.add_train_conf(train_conf)
        else:
            raise OptionNotSupportedError("The requested Experiment class: {} doesn't support the requested training"
                                          " mode: {}".format(self.__class__, train_mode))
