###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  imagenet input pipeline
# Date:         11.2016
#
#
# TODO: 23 images are not jpeg and should be used with the according decoder.

###############################################################################
# NOTE: this code has been modified from the original version of Imanol Schlag
# to be in line with the architecture of this program
#


""" Usage:
import tensorflow as tf
sess = tf.Session()

with tf.device('/cpu:0'):
  from datasets.imagenet import imagenet_data
  d = imagenet_data(batch_size=64, sess=sess)
  image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()

image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
print(image_batch.shape)
print(target_batch.shape)
"""

import tensorflow as tf
import numpy as np
import threading

from input import imagenet_reader as imagenet
from input.data import Data


class ImagenetData(Data):
    """
    Downloads the imagenet dataset and creates an input pipeline ready to be fed into a model.

    memory calculation:
      1 image is 299*299*3*4 bytes = ~1MB
      1024MB RAM = ~1000 images

    empirical memory usage with default config:
      TensorFlow +500MB
      imagenet_utils (loading all paths and labels) +400MB
      build input pipeline and fill queues +2.2GB

    - decodes jpg images
    - scales images into a uniform size
    - shuffles the input if specified
    - builds batches
    """
    NUMBER_OF_CLASSES = 200
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    NUM_OF_CHANNELS = 3

    def __init__(self, general_config,
                 train_dirs: [str],
                 validation_dir: str,
                 extras: [str],
                 batch_queue_capacity=1000,
                 image_height=IMAGE_HEIGHT,
                 image_width=IMAGE_WIDTH):
        """ Downloads the data if necessary. """
        print("Loading imagenet data")
        imagenet.ImagenetReader.set_parameters(train_dirs, validation_dir, extras)
        my_imagenet = imagenet.ImagenetReader.get_data()
        super().__init__(general_config, my_imagenet, image_height, image_width)
        self.batch_queue_capacity = batch_queue_capacity + 3 * self.curr_config.batch_size
        self.data_reader.check_if_downloaded()

    def build_train_data_tensor(self, shuffle=False, augmentation=False):
        img_path, cls = self.data_reader.load_training_data()
        return self.__build_generic_data_tensor(img_path, cls, shuffle, augmentation, testing=False)

    def build_test_data_tensor(self, shuffle=False, augmentation=False):
        img_path, cls = self.data_reader.load_test_data()
        return self.__build_generic_data_tensor(img_path, cls, shuffle, augmentation, testing=True)

    def __build_generic_data_tensor(self, all_img_paths, all_targets, shuffle, augmentation, testing):
        """
        Creates the input pipeline and performs some preprocessing.
        The full dataset needs to fit into memory for this version.
        """

        def load_images(single_path, single_target):
            """
            Maps the paths and labels with the corresponding Tensors that are going to be used as Input for the training
            :param single_path: a path to an image of .jpeg type
            :param single_target: a number that corresponds with the label of the sample
            :return: a tuple with two tensors, the first one represents the image data and the second one represents
            the label.
            """
            # one hot encode the target
            single_target = tf.cast(tf.subtract(single_target, tf.constant(1)), tf.int32)
            single_target = tf.one_hot(single_target, depth=self.NUMBER_OF_CLASSES)

            # load the jpg image according to path
            file_content = tf.read_file(single_path)
            single_image = tf.image.decode_jpeg(file_content, channels=self.NUM_OF_CHANNELS)

            # convert to [0, 1]
            single_image = tf.image.convert_image_dtype(single_image,
                                                        dtype=tf.float32,
                                                        saturate=True)

            single_image = tf.image.resize_images(single_image, [self.IMAGE_HEIGHT, self.IMAGE_WIDTH])

            # Data Augmentation
            if augmentation:
                single_image = tf.image.resize_image_with_crop_or_pad(single_image, self.IMAGE_HEIGHT + 4,
                                                                      self.IMAGE_WIDTH + 4)
                single_image = tf.random_crop(single_image, [self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.NUM_OF_CHANNELS])
                single_image = tf.image.random_flip_left_right(single_image)

                single_image = tf.image.per_image_standardization(single_image)

            return single_image, single_target

        # Creates the dataset
        filenames = tf.constant(all_img_paths)
        labels = tf.constant(all_targets)

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(load_images)

        if shuffle:
            dataset.shuffle(buffer_size=self.batch_queue_capacity, seed=12345)
        dataset = dataset.batch(self.curr_config.batch_size)

        if testing:
            dataset = dataset.repeat()
        else:
            dataset = dataset.repeat(self.curr_config.epochs)

        iterator = dataset.make_one_shot_iterator()
        images_batch, target_batch = iterator.get_next()

        return images_batch, target_batch

    def close(self):
        print("Closing Data pipeline...")
        return
