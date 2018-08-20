import tensorflow as tf
import os
import psutil

from input import cifar_reader as cifar
from input.data import Data


class CifarData(Data):
    NUMBER_OF_CLASSES = 10
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32

    def __init__(self, general_config,
                 train_dirs: [str],
                 validation_dir: str,
                 extras: [str],
                 batch_queue_capacity=1000,
                 image_height=IMAGE_HEIGHT,
                 image_width=IMAGE_WIDTH):
        """ Downloads the data if necessary. """
        print("Loading Cifar 1 0 data...")
        cifar.CifarReader.set_parameters(train_dirs, validation_dir, extras)
        my_cifar = cifar.CifarReader.get_data()
        super().__init__(general_config, my_cifar, image_height, image_width)
        self.data_reader.check_if_downloaded()
        self.batch_queue_capacity = batch_queue_capacity

    def build_train_data_tensor(self, shuffle=False, augmentation=False, skip_count=0):
        filename, _ = self.data_reader.load_training_data()
        return self.__build_generic_data_tensor(filename, shuffle, augmentation, testing=False,
                                                skip_count=skip_count)

    def build_test_data_tensor(self, shuffle=False, augmentation=False):
        filename, _ = self.data_reader.load_test_data()
        return self.__build_generic_data_tensor(filename, shuffle, augmentation, testing=True)

    def change_dataset_part(self, index: int):
        pass

    def __build_generic_data_tensor(self, filename, shuffle, augmentation, testing, skip_count=0):
        
        def parser(serialized_example):
            """Parses a single tf.Example into image and label tensors."""
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

            label = tf.cast(features['label'], tf.int32) 
            label = tf.one_hot(label, depth=self.NUMBER_OF_CLASSES)
            # TODO: preprocessing custom

            return image, label

        process = psutil.Process(os.getpid())
        # TODO: BORRAR LOS TENSORES
        print("memory before build ", process.memory_info().rss)

        # Creates the dataset
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(parser, num_parallel_calls=self.batch_queue_capacity)
        print(dataset)

        print(" Memory after build ", process.memory_info().rss)
        ##filenames = tf.convert_to_tensor(raw_images)
        ##labels = tf.convert_to_tensor(raw_targets)
        # dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
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

    def __del__(self):
        pass

    def close(self):
        pass
