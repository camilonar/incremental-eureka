"""
Module for the data pipeline of MNIST dataset.
"""
import tensorflow as tf

from etl.data import Data
import utils.constants as const
from etl.reader.tfrecords_reader import TFRecordsReader


class FashionMnistData(Data):
    """
    Data pipeline for Fashion-MNIST dataset
    """

    def __init__(self, general_config,
                 train_dirs: [str],
                 validation_dir: str,
                 batch_queue_capacity=10000,
                 image_height=28,
                 image_width=28):
        print("Loading fashion mnist data...")
        my_f_mnist = TFRecordsReader(train_dirs, validation_dir, general_config.train_mode)
        super().__init__(general_config, my_f_mnist, image_height, image_width)
        self.batch_queue_capacity = batch_queue_capacity
        self.data_reader.check_if_data_exists()

    def build_train_data_tensor(self, shuffle=True, augmentation=False, skip_count=0):
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
            """
            Parses a single tf.Example into image and label tensors.

            :param serialized_example: serialized example in tfrecord type
            :return: a tuple with two tensors, the first one represents the image data and the second one represents
                the label.
            """
            number_of_classes = 10
            image_height = 28
            image_width = 28

            features = tf.parse_single_example(
                serialized_example,
                features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'label': tf.FixedLenFeature([], tf.int64),
                    'image_raw': tf.FixedLenFeature([], tf.string)
                })

            image = tf.decode_raw(features['image_raw'], tf.uint8)
            image.set_shape((image_width * image_height))
            # Reshape from [depth * height * width] to [depth, height, width].

            image = tf.cast(
                tf.transpose(tf.reshape(image, [1, image_height, image_width]), [1, 2, 0]),
                tf.float32)

            image = tf.image.convert_image_dtype(image,
                                                 dtype=tf.float32,
                                                 saturate=True) * (1 / 255.0)

            image = tf.image.resize_images(image, [self.image_width, self.image_height])

            label = tf.cast(features['label'], tf.int32)
            label = tf.one_hot(label, depth=number_of_classes)

            return image, label

        # Creates the dataset
        dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=len(self.general_config.train_configurations))
        dataset = dataset.map(parser, num_parallel_calls=8)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.batch_queue_capacity, seed=const.SEED)
        dataset = dataset.batch(self.curr_config.batch_size)
        dataset = dataset.prefetch(self.batch_queue_capacity)

        # Only does multiple epochs if the dataset is going to be used for training
        if not testing:
            dataset = dataset.repeat(self.curr_config.epochs)

        dataset.skip(skip_count)

        iterator = dataset.make_initializable_iterator()
        images_batch, target_batch = iterator.get_next()
        return iterator, images_batch, target_batch

    def close(self):
        pass
