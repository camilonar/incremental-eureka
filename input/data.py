"""
Module for data input.
"""
import tensorflow as tf
from input.reader import Reader


class Data(object):
    """
    This class acts as an interface for data pipelines.

    This structure is based in the pipelines from:
        https://github.com/ischlag/tensorflow-input-pipelines
    """

    def __init__(self, batch_size: int, sess: tf.Session, data_reader: Reader,
                 image_height, image_width):
        """
        Creates a Data pipeline object for a dataset composed of images
        :param batch_size: size of the batches of data
        :param sess: the Tensorflow Session
        :param data_reader: the corresponding Reader of the data of the dataset
        :param image_height: the height at which the images are going to be rescaled
        :param image_width: the width at which the images are going to be rescaled
        """
        self.batch_size = batch_size
        self.sess = sess
        self.data_reader = data_reader
        self.image_height = image_height
        self.image_width = image_width

    def build_train_data_tensor(self, shuffle=False, augmentation=False):
        """
        Builds the training data tensor
        :param shuffle: specifies whether the data is going to be randomly shuffled or not
        :param augmentation: specifies if there is going to be performed a data augmentation process
        :return: a tuple of Tensors, where the first value corresponds with the tensor of training data and the second
        is the tensor with the corresponding labels of the data
        """
        raise NotImplementedError("The subclass hasn't implemented the build_train_data_tensor method")

    def build_test_data_tensor(self, shuffle=False, augmentation=False):
        """
        Builds the test data tensor
        :param shuffle: specifies whether the data is going to be randomly shuffled or not
        :param augmentation: specifies if there is going to be performed a data augmentation process
        :return: a tuple of Tensors, where the first value corresponds with the tensor of test data and the second is
        the tensor with the corresponding labels of the data
        """
        raise NotImplementedError("The subclass hasn't implemented the build_test_data_tensor method")

    def change_dataset_part(self, index: int):
        """
        It changes the target archive of directory from which the training data is being extracted. This ONLY applies
        to the training data and NOT to the test data.
        :param index: the number of the mega-batch, starting from 0. I.e. for the first batch, this would be 0
        :return: None
        """
        self.data_reader.change_dataset_part(index)

    def __del__(self):
        """
        Destroys the object and closes the pipeline
        :return: None
        """
        self.close()

    def close(self):
        """
        Closes the pipeline
        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the close method")
