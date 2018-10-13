"""
Module with the reader for Cifar-10 dataset
"""
import pickle
import numpy as np

from input.reader import Reader

size_image = 32
numero_canales = 3

number_of_classes = 100


def _convert_raw_to_image(raw):
    '''
    convert the images from format cifar100 to one array of 4 dimensions
    [number of images,heigth, width, number of color chanels]
    each pixel represented by one floating number from 0 to 1
    :param raw: array, one-dimensional array that represents the image
    :return: array of 4 dimensions [number of images,heigth, width, number of color chanels]
    '''

    
    raw_float = np.array(raw, dtype=float) / 255.0

    # reshape
    images = raw_float.reshape([-1, numero_canales, size_image, size_image])
    images = images.transpose([0, 2, 3, 1])

    return images


def _to_one_hot(class_numbers, num_classes=None):
    '''
    converts numeric values ​​to onehot arrays
    :param class_numbers: array, values ​​that correspond to the categories of a classification
        values ​​between 0 and the number of classes
    :param num_classes: number of class of dataset
    :return: array 2d;  contains the onehot representation of each of the values in
      class_numbers variable
    '''

    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    one_hot=np.eye(num_classes, dtype=float)[class_numbers]

    return one_hot


def _unpickle(filename):
    '''
    performs the deserialization process of a file
    :param filename: string, Serialized file path
    :return: raw data in bytes
    '''
    print("Loading data: " + filename)
    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data


def _get_human_readable_labels():
    '''
    read from the metadata file the categorization tags for humans
    :return: list of labels in string format for humans
    '''

    raw = _unpickle(filename=Cifar100Reader._metadata_file)[b'label_names']
    humans = [x.decode('utf-8') for x in raw]
    return humans


def _load_batch(filename):
    """
    load file serialized of dataset CIFAR-10
    """

    data = _unpickle(filename)
    # obtaing images in format raw *(serialized)
    raw_image = data[b'data']
    # obtain number of class
    cls = np.array(data[b'labels'])
    images = _convert_raw_to_image(raw_image)
    return images, cls


def _load_data(filename):
    """
    load file and retur
    """
    images, cls = _load_batch(filename)
    return images, cls, _to_one_hot(class_numbers=cls, num_classes=number_of_classes)


class Cifar100Reader(Reader):
    """
    Reader for Cifar-100 dataset
    """
    __train_dirs, __validation_dir, _metadata_file = None, None, None
    data = None

    def reload_training_data(self):
        pass

    def __init__(self, train_dirs: [str], validation_dir: str):
        super().__init__(train_dirs, validation_dir)
        print("TEST PATH ", validation_dir)
        print("TRAIN PATHs ", train_dirs)

    def load_class_names(self):
        return _get_human_readable_labels()

    def load_training_data(self):
        return self.curr_path, None

    def load_test_data(self):
        return self.test_path, None

    @classmethod
    def get_data(cls):
        """
        Gets the data of CIFAR-10. set_parameters must be called before this method or an Exception may be raised.
        :return: a Singleton object of CifarReader
        """
        if not cls.data:
            cls.data = Cifar100Reader(cls.__train_dirs, cls.__validation_dir)
        return cls.data

    @classmethod
    def set_parameters(cls, train_dirs: [str], validation_dir: str, extras: [str]):
        """
        Sets the parameters for the Singleton reader
        :param train_dirs: the paths to the training data
        :param validation_dir: the path to the testing data
        :param extras: an array with extra paths, must be of this form:
                        [metadata_file_path]
        """
        cls.__train_dirs = train_dirs
        cls.__validation_dir = validation_dir
        cls._metadata_file = extras[0]
