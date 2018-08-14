"""
This module has the default paths for the four datasets supported in the base version:
-MNIST
-CIFAR-10
-CALTECH-101
-TINY IMAGENET
"""
import utils.constants as const
from errors import OptionNotSupportedError


# TODO poner los paths por defecto en todos menos MNIST
def __get_mnist_paths():
    """
    It gives the default paths to the training and testing data of MNIST
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, and the second value is a string corresponding to the path of the testing data
    """
    pass


def __get_cifar_paths():
    """
    It gives the default paths to the training and testing data of CIFAR-10
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, and the second value is a string corresponding to the path of the testing data
    """
    base_folder = "../datasets/cifar-10-batches-py"
    base = base_folder + "/data_batch_"
    tr_paths = [base + "1", base + "2", base + "3", base + "4", base + "5"]
    test_path = base_folder + "/test_batch"
    return tr_paths, test_path


def __get_caltech_paths():
    """
    It gives the default paths to the training and testing data of CALTECH-101
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, and the second value is a string corresponding to the path of the testing data
    """
    pass


def __get_tiny_imagenet_paths():
    """
    It gives the default paths to the training and testing data of TINY IMAGENET
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, and the second value is a string corresponding to the path of the testing data
    """
    pass


def __get_not_supported_dataset():
    raise OptionNotSupportedError("The requested dataset doesn't have default paths in the current version of the "
                                  "program.")


def get_paths_from_dataset(dataset: str):
    """
    It gives the default paths to the training and testing data of the supported dataset
    :param dataset: a string representing the dataset
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, and the second value is a string corresponding to the path of the testing data
    """
    dict = {const.DATA_MNIST: __get_mnist_paths,
            const.DATA_CIFAR_10: __get_cifar_paths,
            const.DATA_CALTECH_101: __get_caltech_paths,
            const.DATA_TINY_IMAGENET: __get_tiny_imagenet_paths}
    return dict.get(dataset, __get_not_supported_dataset)()
