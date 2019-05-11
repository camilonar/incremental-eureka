"""
IoC container for the Experiment objects
"""
import utils.constants as const
from experiments.caltech101.caltech_exp_rilbc import CaltechExperimentRILBC
from experiments.caltech256.caltech_256_exp_rms import Caltech256ExperimentRMSProp
from experiments.caltech101.caltech_exp_nil import CaltechExperimentNIL
from experiments.caltech101.caltech_exp_rms import CaltechExperimentRMSProp
from experiments.cifar10.cifar_exp_rilbc import CifarExperimentRILBC
from experiments.cifar100.cifar100_exp_rms import Cifar100ExperimentRMSProp

from experiments.cifar10.cifar_exp_nil import CifarExperimentNIL
from experiments.cifar10.cifar_exp_rms import CifarExperimentRMSProp
from experiments.fashion_mnist.fashion_mnist_exp_nil import FashionMnistExperimentNIL
from experiments.fashion_mnist.fashion_mnist_exp_rilbc import FashionMnistExperimentRILBC
from experiments.fashion_mnist.fashion_mnist_exp_rms import FashionMnistExperimentRMSProp

from experiments.imagenet.imagenet_exp_nil import ImagenetExperimentNIL
from experiments.imagenet.imagenet_exp_rilbc import ImagenetExperimentRILBC
from experiments.imagenet.imagenet_exp_rms import ImagenetExperimentRMSProp

from experiments.mnist.mnist_exp_nil import MnistExperimentNIL
from experiments.mnist.mnist_exp_rilbc import MnistExperimentRILBC
from experiments.mnist.mnist_exp_rms import MnistExperimentRMSProp


class Experiments(object):
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
    testers = {const.TR_BASE: {const.DATA_CALTECH_101: CaltechExperimentRMSProp,
                               const.DATA_CIFAR_10: CifarExperimentRMSProp,
                               const.DATA_TINY_IMAGENET: ImagenetExperimentRMSProp,
                               const.DATA_MNIST: MnistExperimentRMSProp,
                               const.DATA_CALTECH_256: Caltech256ExperimentRMSProp,
                               const.DATA_CIFAR_100: Cifar100ExperimentRMSProp,
                               const.DATA_FASHION_MNIST: FashionMnistExperimentRMSProp
                               },
               const.TR_NIL: {const.DATA_CALTECH_101: CaltechExperimentNIL,
                              const.DATA_CIFAR_10: CifarExperimentNIL,
                              const.DATA_TINY_IMAGENET: ImagenetExperimentNIL,
                              const.DATA_MNIST: MnistExperimentNIL,
                              const.DATA_FASHION_MNIST: FashionMnistExperimentNIL},
               const.TR_RILBC: {const.DATA_CALTECH_101: CaltechExperimentRILBC,
                                const.DATA_CIFAR_10: CifarExperimentRILBC,
                                const.DATA_TINY_IMAGENET: ImagenetExperimentRILBC,
                                const.DATA_MNIST: MnistExperimentRILBC,
                                const.DATA_FASHION_MNIST: FashionMnistExperimentRILBC}
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
