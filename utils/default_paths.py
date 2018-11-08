"""
This module has the default paths for the four datasets supported in the base version:
-MNIST
-CIFAR-10
-CALTECH-101
-TINY IMAGENET

It also contains paths to the weights of some networks that use transfer learning
"""
# TODO actualizar documentación
import os
import utils.constants as const
from errors import OptionNotSupportedError


# TODO revisar todos los paths para asegurar que funcione adecuadamente
def get_alexnet_weights_path():
    """
    It gives the default path of the weights of an AlexNet network previously trained over Imagenet
    :return: a string corresponding to a relative path
    """
    return "../transfer_learning/bvlc_alexnet.npy"


def get_vgg16_weights_path():
    """
    It gives the default path of the weights of an AlexNet network previously trained over Imagenet
    :return: a string corresponding to a relative path
    """
    return "../transfer_learning/vgg16_weights.npz"


def __get_mnist_paths(is_incremental: bool, base_folder: str = const.MNIST_PATH):
    """
    It gives the default paths to the training and testing data of MNIST
    :param is_incremental: True to indicate that the training is gonna contain multiple mega-batches
    :param base_folder: the directory where the datasets are being stored. E.g. if a dataset is stored in
    ../datasets/MNIST, then base_folder should be "../datasets/MNIST"
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data, and the third
    value is an empty array
    """
    base = "train-"
    ext = ".tfrecords"
    name_tr = os.path.join(base_folder, base)
    tr_paths = [name_tr + "{}".format(x) + ext for x in range(1, 6)]

    if not is_incremental:
        # tr_paths = [os.path.join(base_folder, base + "complete" + ext)]
        tr_paths = [tr_paths]

    test_path = os.path.join(base_folder, "validation" + ext)
    return tr_paths, test_path, []


def __get_fashion_mnist_paths(is_incremental: bool, base_folder: str = const.FASHION_MNIST_PATH):
    """
    It gives the default paths to the training and testing data of FASHION MNIST
    :param is_incremental: True to indicate that the training is gonna contain multiple mega-batches
    :param base_folder: the directory where the datasets are being stored. E.g. if a dataset is stored in
    ../datasets/FASHION-MNIST, then base_folder should be "../datasets/FASHION-MNIST"
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data, and the third
    value is an empty array
    """
    base = "train-"
    ext = ".tfrecords"
    name_tr = os.path.join(base_folder, base)
    tr_paths = [name_tr + "{}".format(x) + ext for x in range(1, 6)]

    if not is_incremental:
        tr_paths = [tr_paths]

    test_path = os.path.join(base_folder, "test" + ext)
    return tr_paths, test_path, []


def __get_cifar_paths(is_incremental: bool, base_folder: str = const.CIFAR_10_PATH):
    """
    It gives the default paths to the training and testing data of CIFAR-10
    :param is_incremental: True to indicate that the training is gonna contain multiple mega-batches
    :param base_folder: the directory where the datasets are being stored. E.g. if a dataset is stored in
    ../datasets/cifar10, then base_folder should be "../datasets/cifar10"
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data, and the third
    value is an array of this form: [metadata_file_path]
    """
    base = os.path.join(base_folder, "data_batch_")
    ext = ".tfrecords"
    tr_paths = [base + "{}".format(x) + ext for x in range(1, 6)]

    if not is_incremental:
        tr_paths = [tr_paths]

    test_path = os.path.join(base_folder, "test_batch" + ext)
    metadata_file = os.path.join(base_folder, "batches.meta")
    return tr_paths, test_path, [metadata_file]


# TODO versiones incrementales y no incrementales
def __get_cifar100_paths(is_incremental: bool, base_folder: str = const.CIFAR_100_PATH):
    """
    It gives the default paths to the training and testing data of CIFAR-100
    :param is_incremental: True to indicate that the training is gonna contain multiple mega-batches
    :param base_folder: the directory where the datasets are being stored. E.g. if a dataset is stored in
    ../datasets/cifar-100, then base_folder should be "../datasets/cifar-100"
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data, and the third
    value is an array of this form: [metadata_file_path]
    """
    base = os.path.join(base_folder, "train")
    ext = ".tfrecords"
    tr_paths = [base + ext]

    if not is_incremental:
        tr_paths = [tr_paths]

    test_path = os.path.join(base_folder, "test" + ext)
    metadata_file = os.path.join(base_folder, "meta")
    return tr_paths, test_path, [metadata_file]


def __get_caltech_paths(is_incremental: bool, base_folder: str = const.CALTECH_101_PATH):
    """
    It gives the default paths to the training and testing data of CALTECH-101
    :param is_incremental: True to indicate that the training is gonna contain multiple mega-batches
    :param base_folder: the directory where the datasets are being stored. E.g. if a dataset is stored in
    ../datasets/101_ObjectCategories, then base_folder should be "../datasets/101_ObjectCategories"
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data and the third
    value is an empty array
    """
    base = os.path.join(base_folder, "train")
    if is_incremental:
        paths = [os.path.join(base, "Lote{}".format(x)) for x in range(0, 5)]
    else:
        paths = [base]
    validation_dir = os.path.join(base_folder, "test")
    return paths, validation_dir, []


def __get_caltech_256_paths(is_incremental: bool, base_folder: str = const.CALTECH_256_PATH):
    """
    It gives the default paths to the training and testing data of CALTECH-256
    :param is_incremental: True to indicate that the training is gonna contain multiple mega-batches
    :param base_folder: the directory where the datasets are being stored. E.g. if a dataset is stored in
    ../datasets/256_ObjectCategories, then base_folder should be "../datasets/256_ObjectCategories"
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data and the third
    value is an empty array
    """
    base = os.path.join(base_folder, "train")
    if is_incremental:
        paths = [os.path.join(base, "Lote{}".format(x)) for x in range(0, 5)]
    else:
        paths = [base]
    validation_dir = os.path.join(base_folder, "test")
    return paths, validation_dir, []


def __get_tiny_imagenet_paths(is_incremental: bool, base_folder: str = const.TINY_IMAGENET_PATH):
    """
    It gives the default paths to the training and testing data of TINY IMAGENET
    :param is_incremental: True to indicate that the training is gonna contain multiple mega-batches
    :param base_folder: the directory where the datasets are being stored. E.g. if a dataset is stored in
    ../datasets/tiny-imagenet-200, then base_folder should be "../datasets/tiny-imagenet-200"
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training, the second value is a string corresponding to the path of the testing data and the third
    value is an array of this form: [labels_file_path, metadata_file_path]
    """
    base = os.path.join(base_folder, "train_split")
    if is_incremental:
        train_dirs = [os.path.join(base, "Lote{}".format(x)) for x in range(0, 5)]
    else:
        train_dirs = [base]

    validation_dir = os.path.join(base_folder, "val")
    # validation_dir = train_dirs[0]
    labels_file = os.path.join(base_folder, "wnids.txt")
    metadata_file = os.path.join(base_folder, "words.txt")
    return train_dirs, validation_dir, [labels_file, metadata_file]


def __get_not_supported_dataset(*args):
    raise OptionNotSupportedError("The requested dataset doesn't have default paths in the current version of the "
                                  "program.")


def get_paths_from_dataset(dataset: str, is_incremental: bool, base_folder: str = None):
    """
    It gives the default paths to the training and testing data of the supported dataset
    :param dataset: a string representing the dataset
    :param is_incremental: True to indicate that the training is gonna contain multiple mega-batches
    :param base_folder: the base directory where the dataset is located. If it's not provided, then the default values
    for each dataset will be used
    :return: a tuple, where the first value is an array of strings corresponding to the paths of each one of the
    mega-batches for training (if is_incremental is set to False then this array contains only 1 value corresponding
    to the path to the training data), the second value is a string corresponding to the path of the testing data,
    and the third value is an array of strings corresponding to paths specific for each dataset (e.g. bounding box
    archive path, metadata file path. The third value is an empty array in the case that the requested dataset doesn't
    have dataset-specific paths.
    """
    options = {const.DATA_MNIST: __get_mnist_paths,
               const.DATA_CIFAR_10: __get_cifar_paths,
               const.DATA_CALTECH_101: __get_caltech_paths,
               const.DATA_TINY_IMAGENET: __get_tiny_imagenet_paths,
               const.DATA_CALTECH_256: __get_caltech_256_paths,
               const.DATA_CIFAR_100: __get_cifar100_paths,
               const.DATA_FASHION_MNIST: __get_fashion_mnist_paths
               }
    if base_folder:
        return options.get(dataset, __get_not_supported_dataset)(is_incremental, base_folder)
    else:
        return options.get(dataset, __get_not_supported_dataset)(is_incremental)
