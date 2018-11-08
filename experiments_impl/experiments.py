"""
IoC container for the Experiment objects
"""
import dependency_injector.containers as containers
import dependency_injector.providers as providers

import utils.constants as const
from experiments_impl.caltech_256_rms_exp import Caltech256RMSPropExperiment
from experiments_impl.caltech_dcgan_exp import CaltechDCGANExperiment
from experiments_impl.caltech_rep_exp import CaltechRepExperiment
from experiments_impl.caltech_rms_exp import CaltechRMSPropExperiment
from experiments_impl.cifar_dcgan_exp import CifarDCGANExperiment
from experiments_impl.cifar_rep_exp import CifarRepExperiment
from experiments_impl.cifar_rms_exp import CifarRMSPropExperiment
from experiments_impl.cifar100_rms_exp import Cifar100RMSPropExperiment
from experiments_impl.imagenet_dcgan_exp import ImagenetDCGANExperiment
from experiments_impl.imagenet_rep_exp import ImagenetRepExperiment
from experiments_impl.imagenet_rms_exp import ImagenetRMSPropExperiment
from experiments_impl.mnist_dcgan_exp import MnistDCGANExperiment
from experiments_impl.mnist_rep_exp import MnistRepExperiment
from experiments_impl.mnist_rms_exp import MnistRMSPropExperiment
from experiments_impl.fashion_mnist_rms_exp import FashionMnistRMSPropExperiment


class Experiments(containers.DeclarativeContainer):
    """
    IoC container for the basic experiments included in the framework, which corresponds to datasets:
    -Caltech-101
    -Cifar-10
    -MNIST
    -Tiny Imagenet
    """
    # TODO agregar los otros testers
    testers = {const.TR_BASE: {const.DATA_CALTECH_101: providers.Factory(CaltechRMSPropExperiment),
                               const.DATA_CIFAR_10: providers.Factory(CifarRMSPropExperiment),
                               const.DATA_TINY_IMAGENET: providers.Factory(ImagenetRMSPropExperiment),
                               const.DATA_MNIST: providers.Factory(MnistRMSPropExperiment),
                               const.DATA_CALTECH_256: providers.Factory(Caltech256RMSPropExperiment),
                               const.DATA_CIFAR_100: providers.Factory(Cifar100RMSPropExperiment),
                               const.DATA_FASHION_MNIST: providers.Factory(FashionMnistRMSPropExperiment)
                               },
               const.TR_DCGAN: {const.DATA_CALTECH_101: providers.Factory(CaltechDCGANExperiment),
                                const.DATA_CIFAR_10: providers.Factory(CifarDCGANExperiment),
                                const.DATA_TINY_IMAGENET: providers.Factory(ImagenetDCGANExperiment),
                                const.DATA_MNIST: providers.Factory(MnistDCGANExperiment)},
               const.TR_REP: {const.DATA_CALTECH_101: providers.Factory(CaltechRepExperiment),
                              const.DATA_CIFAR_10: providers.Factory(CifarRepExperiment),
                              const.DATA_TINY_IMAGENET: providers.Factory(ImagenetRepExperiment),
                              const.DATA_MNIST: providers.Factory(MnistRepExperiment)}
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
