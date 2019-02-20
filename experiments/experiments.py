"""
IoC container for the Experiment objects
"""
import dependency_injector.containers as containers
import dependency_injector.providers as providers

import utils.constants as const
from experiments.caltech256.caltech_256_exp_rms import Caltech256ExperimentRMSProp
from experiments.caltech101.caltech_exp_dcgan import CaltechExperimentDCGAN
from experiments.caltech101.caltech_exp_rep import CaltechExperimentRep
from experiments.caltech101.caltech_exp_rms import CaltechExperimentRMSProp
from experiments.cifar100.cifar100_exp_rms import Cifar100ExperimentRMSProp

from experiments.cifar10.cifar_exp_dcgan import CifarExperimentDCGAN
from experiments.cifar10.cifar_exp_rep import CifarExperimentRep
from experiments.cifar10.cifar_exp_rms import CifarExperimentRMSProp
from experiments.fashion_mnist.fashion_mnist_exp_rep import FashionMnistExperimentRep
from experiments.fashion_mnist.fashion_mnist_exp_rms import FashionMnistExperimentRMSProp

from experiments.imagenet.imagenet__exp_dcgan import ImagenetExperimentDCGAN
from experiments.imagenet.imagenet_exp_rep import ImagenetExperimentRep
from experiments.imagenet.imagenet_exp_rms import ImagenetExperimentRMSProp

from experiments.mnist.mnist_exp_dcgan import MnistExperimentDCGAN
from experiments.mnist.mnist_exp_rep import MnistExperimentRep
from experiments.mnist.mnist_exp_rms import MnistExperimentRMSProp


class Experiments(containers.DeclarativeContainer):
    """
    IoC container for the basic experiments included in the framework, which corresponds to datasets:

        - Caltech-101
        - CIFAR-10
        - CIFAR-100
        - MNIST
        - Tiny Imagenet
        - Caltech 256
        - Fashion MNIST

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
                              const.DATA_MNIST: providers.Factory(MnistExperimentRep),
                              const.DATA_FASHION_MNIST: providers.Factory(FashionMnistExperimentRep)}
               }

    @classmethod
    def get_experiment(cls, str_trainer: str, str_dataset: str):
        """
        Gets an Experiment object factory

        :param str_trainer: a string representing the trainer/optimizer
        :param str_dataset: a string representing the dataset
        :return: a Factory provider for the desired trainer and dataset
        :rtype: Factory
        """
        return cls.testers[str_trainer][str_dataset]
