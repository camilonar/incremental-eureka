"""
IoC container for the tester objects
"""
import dependency_injector.containers as containers
import dependency_injector.providers as providers

import utils.constants as const
from tests_impl.caltech_ceal_tester import CaltechCEALTester
from tests_impl.caltech_rep_tester import CaltechRepTester
from tests_impl.caltech_rms_tester import CaltechRMSPropTester
from tests_impl.cifar_ceal_tester import CifarCEALTester
from tests_impl.cifar_rep_tester import CifarRepTester
from tests_impl.cifar_rms_tester import CifarRMSPropTester
from tests_impl.imagenet_ceal_tester import ImagenetCEALTester
from tests_impl.imagenet_rep_tester import ImagenetRepTester
from tests_impl.imagenet_rms_tester import ImagenetRMSPropTester
from tests_impl.mnist_ceal_tester import MnistCEALTester
from tests_impl.mnist_rep_tester import MnistRepTester
from tests_impl.mnist_rms_tester import MnistRMSPropTester


class Testers(containers.DeclarativeContainer):
    """
    IoC container for the basic testers included in the framework, which corresponds to datasets:
    -Caltech-101
    -Cifar-10
    -MNIST
    -Tiny Imagenet
    """
    # TODO agregar los otros testers
    testers = {const.TR_BASE: {const.DATA_CALTECH_101: providers.Factory(CaltechRMSPropTester),
                               const.DATA_CIFAR_10: providers.Factory(CifarRMSPropTester),
                               const.DATA_TINY_IMAGENET: providers.Factory(ImagenetRMSPropTester),
                               const.DATA_MNIST: providers.Factory(MnistRMSPropTester)},
               const.TR_CEAL: {const.DATA_CALTECH_101: providers.Factory(CaltechCEALTester),
                               const.DATA_CIFAR_10: providers.Factory(CifarCEALTester),
                               const.DATA_TINY_IMAGENET: providers.Factory(ImagenetCEALTester),
                               const.DATA_MNIST: providers.Factory(MnistCEALTester)},
               const.TR_REP: {const.DATA_CALTECH_101: providers.Factory(CaltechRepTester),
                              const.DATA_CIFAR_10: providers.Factory(CifarRepTester),
                              const.DATA_TINY_IMAGENET: providers.Factory(ImagenetRepTester),
                              const.DATA_MNIST: providers.Factory(MnistRepTester)}
               }

    @classmethod
    def get_tester(cls, str_trainer: str, str_dataset: str):
        """
        Gets a tester object factory
        :param str_trainer: a string representing the trainer/optimizer
        :param str_dataset: a string representing the dataset
        :return: a Factory provider for the desired trainer and dataset
        """
        return cls.testers[str_trainer][str_dataset]
