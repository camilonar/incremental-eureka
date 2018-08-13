###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  imagenet input pipeline
# Date:         11.2016
#
#

###############################################################################
# NOTE: this code has been modified from the original version of Imanol Schlag
# to be in line with the architecture of this program
#


import tensorflow as tf
import input.mnist_reader as mnist

from input.data import Data


# TODO adaptar al modelo de Dataset
class MnistData(Data):
    """
    Data pipeline for MNIST dataset
    """
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    NUM_THREADS = 8
    NUM_OF_CHANNELS = 3

    def __init__(self, general_config,
                 batch_queue_capacity=1000,
                 image_height=IMAGE_HEIGHT,
                 image_width=IMAGE_WIDTH):
        """ Downloads the data if necessary. """
        print("Loading mnist data...")
        my_mnist = mnist.MnistReader.get_data()
        super().__init__(general_config, my_mnist, image_height, image_width)
        self.batch_queue_capacity = batch_queue_capacity + 3 * self.curr_config.batch_size
        self.data_reader.check_if_downloaded()

    def build_train_data_tensor(self, shuffle=False, augmentation=False):
        raw_images, raw_labels = self.data_reader.load_training_data()
        return self.__build_generic_data_tensor(raw_images, raw_labels, shuffle, augmentation)

    def build_test_data_tensor(self, shuffle=False, augmentation=False):
        raw_images_test, raw_labels_test = self.data_reader.load_test_data()
        return self.__build_generic_data_tensor(raw_images_test, raw_labels_test, shuffle, augmentation)

    def __build_generic_data_tensor(self, images_raw, labels_raw, shuffle, augmentation):
        images_tensor = tf.convert_to_tensor(images_raw)
        targets_tensor = tf.convert_to_tensor(labels_raw)
        images_tensor = tf.reshape(images_tensor, [images_raw.shape[0], 28, 28, 1])
        image, label = tf.train.slice_input_producer([images_tensor, targets_tensor], shuffle=shuffle)

        if augmentation:
            image = tf.image.resize_image_with_crop_or_pad(image, self.IMAGE_HEIGHT + 4, self.IMAGE_WIDTH + 4)
            image = tf.random_crop(image, [self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.NUM_OF_CHANNELS])
            image = tf.image.random_flip_left_right(image)

        image = tf.image.per_image_standardization(image)
        images_batch, labels_batch = tf.train.batch([image, label], batch_size=self.curr_config.batch_size,
                                                    num_threads=self.NUM_THREADS)

        return images_batch, labels_batch

    def close(self):
        print("Closing Data pipeline...")
        return
