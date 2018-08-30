"""
Module for the data pipeline of MNIST dataset.
"""
import tensorflow as tf
from input import mnist_reader as mnist

from input.data import Data


class MnistData(Data):
    """
    Data pipeline for MNIST dataset
    """
    NUMBER_OF_CLASSES = 10
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    IMAGE_WIDTH_RESIZE = 32
    IMAGE_HEIGHT_RESIZE = 32

    def __init__(self, general_config,
                 train_dirs: [str],
                 validation_dir: str,
                 extras: [str],
                 batch_queue_capacity=1000,
                 image_height=IMAGE_HEIGHT,
                 image_width=IMAGE_WIDTH):
        """ Downloads the data if necessary. """
        print("Loading mnist data...")
        mnist.MnistReader.set_parameters(train_dirs, validation_dir, extras)
        my_mnist = mnist.MnistReader.get_data()
        super().__init__(general_config, my_mnist, image_height, image_width)
        self.batch_queue_capacity = batch_queue_capacity
        self.data_reader.check_if_downloaded()

    def build_train_data_tensor(self, shuffle=False, augmentation=False, skip_count=0):
        filename, _ = self.data_reader.load_training_data()
        return self.__build_generic_data_tensor(filename, shuffle, augmentation, testing=False,
                                                skip_count=skip_count)

    def build_test_data_tensor(self, shuffle=False, augmentation=False):
        filename, _ = self.data_reader.load_test_data()
        return self.__build_generic_data_tensor(filename, shuffle, augmentation, testing=True)

    def __build_generic_data_tensor(self, filename, shuffle, augmentation, testing, skip_count=0):
        """
         Creates the input pipeline and performs some preprocessing.
        """
        def parser(serialized_example):
            '''
            Parses a single tf.Example into image and label tensors.
            :param serialized_example: serialized example in tfrecord type
            :return: a tuple with two tensors, the first one represents the image data and the second one represents
            the label.
            '''

            features = tf.parse_single_example(
                serialized_example,
                features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'depth': tf.FixedLenFeature([], tf.int64),
                    'label': tf.FixedLenFeature([], tf.int64),
                    'image_raw': tf.FixedLenFeature([], tf.string)
                })

            image = tf.decode_raw(features['image_raw'], tf.int32)
            image.set_shape((self.IMAGE_WIDTH * self.IMAGE_HEIGHT))
            # Reshape from [depth * height * width] to [depth, height, width].

            image = tf.cast(
                tf.transpose(tf.reshape(image, [1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH]), [1, 2, 0]),
                tf.float32)

            image = tf.image.resize_images(image, [self.IMAGE_WIDTH_RESIZE, self.IMAGE_HEIGHT_RESIZE])

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
        iterator = dataset.make_one_shot_iterator()
        images_batch, target_batch = iterator.get_next()
        return images_batch, target_batch

    def close(self):
        pass
