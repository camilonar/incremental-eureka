"""
IoC container for the Experiment objects
"""
import dependency_injector.containers as containers
import dependency_injector.providers as providers

import utils.constants as const
from experiments_impl.caltech_256_exp_rms import Caltech256ExperimentRMSProp
from experiments_impl.caltech_exp_dcgan import CaltechExperimentDCGAN
from experiments_impl.caltech_exp_rep import CaltechExperimentRep
from experiments_impl.caltech_exp_rms import CaltechExperimentRMSProp
from experiments_impl.cifar100_exp_rms import Cifar100ExperimentRMSProp

from experiments_impl.cifar_exp_dcgan import CifarExperimentDCGAN
from experiments_impl.cifar_exp_rep import CifarExperimentRep
from experiments_impl.cifar_exp_rms import CifarExperimentRMSProp
from experiments_impl.fashion_mnist_exp_rms import FashionMnistExperimentRMSProp

from experiments_impl.imagenet__exp_dcgan import ImagenetExperimentDCGAN
from experiments_impl.imagenet_exp_rep import ImagenetExperimentRep
from experiments_impl.imagenet_exp_rms import ImagenetExperimentRMSProp

from experiments_impl.mnist_exp_dcgan import MnistExperimentDCGAN
from experiments_impl.mnist_exp_rep import MnistExperimentRep
from experiments_impl.mnist_exp_rms import MnistExperimentRMSProp


class Experiments(containers.DeclarativeContainer):
    """
    IoC container for the basic experiments included in the framework, which corresponds to datasets:
    -Caltech-101
    -Cifar-10
    -MNIST
    -Tiny Imagenet
    """
    # TODO agregar los otros testers
    testers = {const.TR_BASE: {const.DATA_CALTECH_101: providers.Factory(CaltechExperimentRMSProp),
                               const.DATA_CIFAR_10: providers.Factory(CifarExperimentRMSProp),
                               const.DATA_TINY_IMAGENET: providers.Factory(ImagenetExperimentRMSProp),
                               const.DATA_MNIST: providers.Factory(MnistExperimentRMSProp),
                               const.DATA_CALTECH_256: providers.Factory(Caltech256ExperimentRMSProp),
                               const.DATA_CIFAR_100: providers.Factory(Cifar100ExperimentRMSProp),
                               const.DATA_FASHION_MNIST: providers.Factory(FashionMnistExperimentRMSProp)
                               },
               const.TR_DCGAN: {const.DATA_CALTECH_101: providers.Factory(CaltechExperimentDCGAN),
                                const.DATA_CIFAR_10: providers.Factory(CifarExperimentDCGAN),
                                const.DATA_TINY_IMAGENET: providers.Factory(ImagenetExperimentDCGAN),
                                const.DATA_MNIST: providers.Factory(MnistExperimentDCGAN)},
               const.TR_REP: {const.DATA_CALTECH_101: providers.Factory(CaltechExperimentRep),
                              const.DATA_CIFAR_10: providers.Factory(CifarExperimentRep),
                              const.DATA_TINY_IMAGENET: providers.Factory(ImagenetExperimentRep),
                              const.DATA_MNIST: providers.Factory(MnistExperimentRep)}
               }

    @classmethod
    def get_experiment(cls, str_trainer: str, str_dataset: str):
        """
        Gets an Experiment object factory
        :param str_trainer: a string representing the trainer/optimizer
        :param str_dataset: a string representing the dataset
        :return: a Factory provider for the desired trainer and dataset
        """
        return cls.testers[str_trainer][str_dataset]
