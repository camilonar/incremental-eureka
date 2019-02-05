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

from input.data import Data
import utils.constants as const
from input.reader.directory_reader import DirectoryReader


class ImagenetData(Data):
    """
    Creates an input pipeline for Tiny Imagenet ready to be fed into a model.

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

    def __init__(self, general_config,
                 train_dirs: [str],
                 validation_dir: str,
                 batch_queue_capacity=1000,
                 image_height=256,
                 image_width=256):
        print("Loading imagenet data")
        my_imagenet = DirectoryReader(train_dirs, validation_dir)
        super().__init__(general_config, my_imagenet, image_height, image_width)
        self.batch_queue_capacity = batch_queue_capacity + 3 * self.curr_config.batch_size
        self.data_reader.check_if_data_exists()

    def build_train_data_tensor(self, shuffle=False, augmentation=False, skip_count=0):
        img_path, cls = self.data_reader.load_training_data()
        return self.__build_generic_data_tensor(img_path, cls, shuffle, augmentation, testing=False,
                                                skip_count=skip_count)

    def build_test_data_tensor(self, shuffle=False, augmentation=False):
        img_path, cls = self.data_reader.load_test_data()
        return self.__build_generic_data_tensor(img_path, cls, shuffle, augmentation, testing=True)

    def __build_generic_data_tensor(self, all_img_paths, all_targets, shuffle, augmentation, testing, skip_count=0):
        """
        Creates the input pipeline and performs some preprocessing.
        The full dataset needs to fit into memory for this version.
        """
        number_of_classes = 200
        num_of_channels = 3

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
            single_target = tf.one_hot(single_target, depth=number_of_classes)

            # load the jpg image according to path
            file_content = tf.read_file(single_path)
            image = tf.image.decode_jpeg(file_content, channels=num_of_channels)

            # convert to [0, 1]
            image = tf.image.convert_image_dtype(image,
                                                 dtype=tf.float32,
                                                 saturate=True)

            image = tf.image.resize_images(image, [self.image_height, self.image_width])

            # Data Augmentation
            if augmentation:
                distorted_image = tf.random_crop(image, [self.image_height, self.image_width, 3])
                # Randomly flip the image horizontally.
                distorted_image = tf.image.random_flip_left_right(distorted_image)
                # Because these operations are not commutative, consider randomizing
                # the order their operation.
                # NOTE: since per_image_standardization zeros the mean and makes
                # the stddev unit, this likely has no effect see tensorflow#1458.
                distorted_image = tf.image.random_brightness(distorted_image,
                                                             max_delta=63)
                distorted_image = tf.image.random_contrast(distorted_image,
                                                           lower=0.2, upper=1.8)
                # Subtract off the mean and divide by the variance of the pixels.
                # image = tf.image.per_image_standardization(distorted_image)

                # Set the shapes of tensors.
                image.set_shape([self.image_height, self.image_width, 3])

            return image, single_target

        # Creates the dataset
        filenames = tf.constant(all_img_paths)
        labels = tf.constant(all_targets)

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(load_images)

        if shuffle:
            dataset.shuffle(buffer_size=self.batch_queue_capacity, seed=const.SEED)
        dataset = dataset.batch(self.curr_config.batch_size)

        if not testing:
            dataset = dataset.repeat(self.curr_config.epochs)

        dataset.skip(skip_count)

        iterator = dataset.make_initializable_iterator()
        images_batch, target_batch = iterator.get_next()

        return iterator, images_batch, target_batch

    def close(self):
        print("Closing Data pipeline...")
        return
