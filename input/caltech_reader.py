"""
Module for the reader of Caltech-101 dataset.
It reads the dataset from a number of directories previously created and gives a list of filenames and labels.
"""
import random
import os

from input.reader import Reader

valid_ext = [".jpg", ".gif", ".png", ".jpeg"]


###############################################################################
# Some TensorFlow Inception functions (ported to python3)
# source: https://github.com/tensorflow/models/blob/master/inception/inception/data/build_imagenet_data.py

def _find_image_files(path, categories):
    '''
    :param path: directory where the images are located.
    :param categories: list of strings that contain the caltech dataset categories
    :return: a tuple with two lists, the first one represents the paths of images data and the second one is a integer thats
    represents the correct category of the image.
    '''

    filenames = []
    labels = []
    # load all images in folder path 
    for i, category in enumerate(categories):
        print("LOAD CATEGORY", category)
        for f in os.listdir(path + "/" + category):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_ext:
                continue
            fullpath = os.path.join(path + "/" + category, f)
            filenames.append(fullpath) 
            label_curr = i
            labels.append(label_curr)
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)
    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print("Number of filenames: {}".format(len(filenames)))
    print("Number of labels: {}".format(len(labels)))
    ncategories = len(categories)
    print(ncategories)

    return filenames, labels


class CaltechReader(Reader):
    """
    Reader for Caltech-101 dataset
    """
    __train_dirs, __validation_dir = None, None
    data = None

    def __init__(self, train_dirs: [str], validation_dir: str):
        """
        Creates an CaltechReader object
        :param train_dirs: the paths to the training data
        :param validation_dir: the path to the testing data
        """
        super().__init__(train_dirs, validation_dir)
        self.categories = sorted(os.listdir(self.curr_path))
        self.val_filenames, self.val_labels = _find_image_files(validation_dir, self.categories)
        self.train_filenames, self.train_labels = _find_image_files(self.curr_path, self.categories)

    def load_class_names(self):
        return self.categories

    def load_training_data(self):
        return self.train_filenames, self.train_labels

    def load_test_data(self):
        return self.val_filenames, self.val_labels

    @classmethod
    def get_data(cls):
        """
        Gets the data of Caltech-101. set_parameters must be called before this method or an Exception may be raised.
        :return: a Singleton object of CaltechReader
        """
        if not cls.data:
            cls.data = CaltechReader(cls.__train_dirs, cls.__validation_dir)
        return cls.data

    @classmethod
    def set_parameters(cls, train_dirs: [str], validation_dir: str):
        """
        Sets the parameters for the Singleton reader
        :param train_dirs: the paths to the training data
        :param validation_dir: the path to the testing data
        """
        cls.__train_dirs = train_dirs
        cls.__validation_dir = validation_dir

    def reload_training_data(self):
        self.train_filenames, self.train_labels = _find_image_files(self.curr_path, self.categories)
