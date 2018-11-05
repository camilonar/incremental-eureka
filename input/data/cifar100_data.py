"""
Module for the data pipeline of Cifar-10 dataset
"""
import tensorflow as tf

from input.reader import cifar100_reader as cifar100
from input.data import Data


class Cifar100Data(Data):
    """
    Data pipeline for Cifar-100
    """
    NUMBER_OF_CLASSES = 100
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    IMAGE_WIDTH_RESIZE = 224
    IMAGE_HEIGHT_RESIZE = 224

    def __init__(self, general_config,
                 train_dirs: [str],
                 validation_dir: str,
                 extras: [str],
                 batch_queue_capacity=1000,
                 image_height=IMAGE_HEIGHT,
                 image_width=IMAGE_WIDTH):
        """ Downloads the data if necessary. """
        print("Loading Cifar-100 data...")
        cifar100.Cifar100Reader.set_parameters(train_dirs, validation_dir, extras)
        my_cifar = cifar100.Cifar100Reader.get_data()
        super().__init__(general_config, my_cifar, image_height, image_width)
        self.data_reader.check_if_downloaded()
        self.batch_queue_capacity = batch_queue_capacity

    def build_train_data_tensor(self, shuffle=False, augmentation=True, skip_count=0):
        filename, _ = self.data_reader.load_training_data()
        return self.__build_generic_data_tensor(filename, shuffle, augmentation, testing=False,
                                                skip_count=skip_count)

    def build_test_data_tensor(self, shuffle=False, augmentation=False):
        filename, _ = self.data_reader.load_test_data()
        return self.__build_generic_data_tensor(filename, shuffle, augmentation, testing=True)

    def __build_generic_data_tensor(self, filename, shuffle, augmentations, testing, skip_count=0):
        """
        Creates the input pipeline and performs some preprocessing.

        """

        def parser(serialized_example):
            """
            Parses a single tf.Example into image and label tensors.
            :param serialized_example: serialized example in tfrecord type
            :return: a tuple with two tensors, the first one represents the image data and the second one represents
            the label.
            """

            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)
                })

            image = tf.decode_raw(features['image'], tf.uint8)
            image.set_shape([3 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH])

            # Reshape from [depth * height * width] to [depth, height, width].
            image = tf.cast(
                tf.transpose(tf.reshape(image, [3, self.IMAGE_HEIGHT, self.IMAGE_WIDTH]), [1, 2, 0]),
                tf.float32)

            image = tf.image.convert_image_dtype(image,
                                                 dtype=tf.float32,
                                                 saturate=True) * (1 / 255.0)
            # Data Augmentation
            if augmentations:
                distorted_image = tf.random_crop(image, [self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3])
                # Randomly flip the image horizontally.
                distorted_image = tf.image.random_flip_up_down(distorted_image)
                distorted_image = tf.image.random_flip_left_right(distorted_image)
                # Because these operations are not commutative, consider randomizing
                # the order their operation.
                # NOTE: since per_image_standardization zeros the mean and makes
                # the stddev unit, this likely has no effect see tensorflow#1458.

                distorted_image = tf.image.random_contrast(distorted_image,
                                                           lower=0.2, upper=1.8)
                # Subtract off the mean and divide by the variance of the pixels.
                image = tf.image.per_image_standardization(distorted_image)

                # Set the shapes of tensors.
                image.set_shape([self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3])

            image = tf.image.resize_images(image, [self.IMAGE_WIDTH_RESIZE, self.IMAGE_HEIGHT_RESIZE])
            image = tf.image.per_image_standardization(image)

            label = tf.cast(features['label'], tf.int32)
            label = tf.one_hot(label, depth=self.NUMBER_OF_CLASSES)

            return image, label

        # Creates the dataset
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(parser, num_parallel_calls=self.batch_queue_capacity)

        if shuffle:
            dataset.shuffle(buffer_size=self.batch_queue_capacity, seed=12345)

        dataset = dataset.batch(self.curr_config.batch_size)
        # Only does multiple epochs if the dataset is going to be used for training
        if not testing:
            dataset = dataset.repeat(self.curr_config.epochs)

        dataset.skip(skip_count)

        iterator = dataset.make_initializable_iterator()
        images_batch, target_batch = iterator.get_next()

        return iterator, images_batch, target_batch

    def close(self):
        pass