import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt

from input.reader import Reader
from tensorflow.examples.tutorials.mnist import input_data

# number of lots to divide the dataset
split_n = 5

# TODO mejorar: hay funciones que van a ser depreciadas y esta desorganizado
class MnistReader(Reader):
    """
        Reader for Caltech101 dataset
    """
    data = None

    def __init__(self):
        self.mnist = input_data.read_data_sets("../datasets/MNIST_data/", one_hot=True)

    def load_training_data(self):
        return self.mnist.train.images, self.mnist.train.labels

    def load_test_data(self):
        return self.mnist.test.images, self.mnist.test.labels

    def load_class_names(self):
        pass

    def change_dataset_part(self, index: int):
        pass

    def reload_training_data(self):
        pass

    @classmethod
    def get_data(cls):
        """
        Gets the data of Imagenet
        :return: a Singleton object of ImagenetReader
        """
        if not cls.data:
            cls.data = MnistReader()
        return cls.data

    def _printNimages(self):
        first_array = self.mnist.train.images[0].reshape(28, 28)
        plt.imshow(first_array)
        # Actually displaying the plot if you are not in interactive mode
        plt.show()
