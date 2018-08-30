"""
This module has the default paths for the four datasets supported in the base version:
-MNIST
-CIFAR-10
-CALTECH-101
-TINY IMAGENET
"""
import utils.constants as const
from errors import OptionNotSupportedError


def __get_mnist_paths():
    """
    It gives the default paths to the training and testing data of MNIST
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data, and the third
    value is an empty array
    """
    path = "../datasets/MNIST_data/"
    base = "train-"
    ext = ".tfrecords"
    name_tr = path + base
    tr_paths = [name_tr + "{}".format(x) + ext for x in range(1, 6)]
    test_path = path + 'test.' + ext
    return tr_paths, test_path, []


def __get_cifar_paths():
    """
    It gives the default paths to the training and testing data of CIFAR-10
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data, and the third
    value is an array of this form: [metadata_file_path]
    """
    base_folder = "../datasets/cifar-10-batches-py"
    base = base_folder + "/data_batch_"
    ext = ".tfrecords"
    tr_paths = [base + "{}.".format(x) + ext for x in range(1, 6)]
    test_path = base_folder + "/test_batch" + ext
    metadata_file = base_folder + "/batches.meta"
    return tr_paths, test_path, metadata_file


def __get_caltech_paths():
    """
    It gives the default paths to the training and testing data of CALTECH-101
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data and the third
    value is an empty array
    """
    base_folder = "../datasets/101_ObjectCategories_split/"
    base = base_folder + "train/"
    paths = [base + "Lote{}/".format(x) for x in range(0, 5)]
    validation_dir = base_folder + "val/"
    return paths, validation_dir, []


def __get_tiny_imagenet_paths():
    """
    It gives the default paths to the training and testing data of TINY IMAGENET
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data and the third
    value is an array of this form: [labels_file_path, metadata_file_path]
    """
    base_folder = "../datasets/tiny-imagenet-200/"
    base = base_folder + "train_split/"
    train_dirs = [base + "Lote{}/".format(x) for x in range(0, 5)]
    validation_dir = base_folder + "val/"
    labels_file = base_folder + "wnids.txt"
    metadata_file = base_folder + "words.txt"
    return train_dirs, validation_dir, [labels_file, metadata_file]


def __get_not_supported_dataset():
    raise OptionNotSupportedError("The requested dataset doesn't have default paths in the current version of the "
                                  "program.")


def get_paths_from_dataset(dataset: str):
    """
    It gives the default paths to the training and testing data of the supported dataset
    :param dataset: a string representing the dataset
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data, and the third
    value is an array of strings corresponding to paths specific for each dataset (e.g. bounding box archive path,
    metadata file path. The third value is an empty array in the case that the requested dataset doesn't have
    dataset-specific paths.
    """
    options = {const.DATA_MNIST: __get_mnist_paths,
               const.DATA_CIFAR_10: __get_cifar_paths,
               const.DATA_CALTECH_101: __get_caltech_paths,
               const.DATA_TINY_IMAGENET: __get_tiny_imagenet_paths}
    return options.get(dataset, __get_not_supported_dataset)()
