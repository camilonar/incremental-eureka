"""
This module has the default paths for the four datasets supported in the base version:
-MNIST
-CIFAR-10
-CALTECH-101
-TINY IMAGENET
"""
import utils.constants as const
from errors import OptionNotSupportedError


# TODO poner los paths por defecto definitivos
def __get_mnist_paths():
    """
    It gives the default paths to the training and testing data of MNIST
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data, and the third
    value is an empty array
    """
    path = "../datasets/MNIST_data/"
    return path, path, []


def __get_cifar_paths():
    """
    It gives the default paths to the training and testing data of CIFAR-10
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data, and the third
    value is an array of this form: [metadata_file_path]
    """
    base_folder = "../datasets/cifar-10-batches-py"
    base = base_folder + "/data_batch_"
    tr_paths = [base + "1.tfrecords", base + "2.tfrecords", base + "3.tfrecords", base + "4.tfrecords", base + "5.tfrecords"]
    test_path = base_folder + "/test_batch.tfrecords"
    metadata_file = base_folder + "/batches.meta"
    return tr_paths, test_path, metadata_file


def __get_caltech_paths():
    """
    It gives the default paths to the training and testing data of CALTECH-101
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data and the third
    value is an empty array
    """
    path = "../datasets/101_ObjectCategories"
    return [path], path, []


def __get_tiny_imagenet_paths():
    """
    It gives the default paths to the training and testing data of TINY IMAGENET
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data and the third
    value is an array of this form: [labels_file_path, metadata_file_path, bounding_box_file]
    """
    train_dirs = ["../datasets/tiny-imagenet-200/train/", "../datasets/tiny-imagenet-200/val/"]
    validation_dir = "../datasets/tiny-imagenet-200/val/"
    labels_file = "../datasets/tiny-imagenet-200/wnids.txt"
    metadata_file = "../datasets/tiny-imagenet-200/words.txt"
    bounding_box_file = "../datasets/tiny-imagenet-200/bounding_boxes.xml"
    return train_dirs, validation_dir, [labels_file, metadata_file, bounding_box_file]


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
    dict = {const.DATA_MNIST: __get_mnist_paths,
            const.DATA_CIFAR_10: __get_cifar_paths,
            const.DATA_CALTECH_101: __get_caltech_paths,
            const.DATA_TINY_IMAGENET: __get_tiny_imagenet_paths}
    return dict.get(dataset, __get_not_supported_dataset)()
