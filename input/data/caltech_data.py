"""
Module for the data pipeline of Caltech-101 dataset
"""

import tensorflow as tf

from input.data import Data
import utils.constants as const
from input.reader.directory_reader import DirectoryReader


class CaltechData(Data):
    """
    Data pipeline for Caltech-101 dataset
    """

    def __init__(self, general_config,
                 train_dirs: [str],
                 validation_dir: [str],
                 batch_queue_capacity=1000,
                 image_height=227,
                 image_width=227):

        """ Downloads the data if necessary. """
        print("Loading Caltech data...")
        my_caltech = DirectoryReader(train_dirs, validation_dir, general_config.train_mode)
        super().__init__(general_config, my_caltech, image_height, image_width)
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
        number_of_classes = 101
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
            single_target = tf.cast(single_target, tf.int32)
            single_target = tf.one_hot(single_target, depth=number_of_classes)

            # load the jpg image according to path
            file_content = tf.read_file(single_path)
            single_image = tf.image.decode_jpeg(file_content, channels=num_of_channels)

            # convert to [0, 1]
            single_image = tf.image.convert_image_dtype(single_image,
                                                        dtype=tf.float32,
                                                        saturate=True)

            single_image = tf.image.resize_images(single_image, [self.image_height, self.image_width])

            # Data Augmentation
            if augmentation:
                single_image = tf.image.resize_image_with_crop_or_pad(single_image, self.image_height + 4,
                                                                      self.image_width + 4)
                single_image = tf.random_crop(single_image, [self.image_height, self.image_width, num_of_channels])
                single_image = tf.image.random_flip_left_right(single_image)

            single_image = tf.image.per_image_standardization(single_image)

            return single_image, single_target

        # Creates the dataset
        filenames = tf.constant(all_img_paths)
        labels = tf.constant(all_targets)

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        dataset = dataset.map(load_images)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.batch_queue_capacity, seed=const.SEED)
        dataset = dataset.batch(self.curr_config.batch_size)

        # Only does multiple epochs if the dataset is going to be used for training
        if not testing:
            dataset = dataset.repeat(self.curr_config.epochs)

        dataset.skip(skip_count)

        iterator = dataset.make_initializable_iterator()
        images_batch, target_batch = iterator.get_next()

        return iterator, images_batch, target_batch

    def close(self):
        return
