
import tensorflow as tf
import numpy as np
import threading

import caltech_reader as caltech
from input.data import Data


class CaltechData(Data):
    NUM_THREADS = 8
    NUMBER_OF_CLASSES = 200
    TRAIN_SET_SIZE = len(caltech.CaltechReader.get_data().train_filenames)  # 1281167 # ~250MB for string with paths
    TEST_SET_SIZE = len(caltech.CaltechReader.get_data().val_filenames)  # 50000
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    NUM_OF_CHANNELS = 3

    def __init__(self, batch_size, sess,
                 filename_feed_size=200,
                 filename_queue_capacity=800,
                 batch_queue_capacity=1000,
                 min_after_dequeue=1000,
                 image_height=IMAGE_HEIGHT,
                 image_width=IMAGE_WIDTH):


        """ Downloads the data if necessary. """
        print("Loading caltech data...")
        my_caltech = caltech.CaltechReader.get_data()
        super().__init__(batch_size, sess, my_caltech, image_height, image_width)
        self.filename_feed_size = filename_feed_size
        self.filename_queue_capacity = filename_queue_capacity
        self.batch_queue_capacity = batch_queue_capacity + 3 * batch_size
        self.min_after_dequeue = min_after_dequeue
        self.data_reader.check_if_downloaded()

    def build_train_data_tensor(self, shuffle=False, augmentation=False):
        img_path, cls = self.data_reader.load_training_data()
        return self.__build_generic_data_tensor(img_path, cls, shuffle, augmentation)

    def build_test_data_tensor(self, shuffle=False, augmentation=False):
        img_path, cls = self.data_reader.load_test_data()
        return self.__build_generic_data_tensor(img_path, cls, shuffle, augmentation)

    def __build_generic_data_tensor(self, all_img_paths, all_targets, shuffle, augmentation):
        """
        Creates the input pipeline and performs some preprocessing.
        The full dataset needs to fit into memory for this version.
        """

        ## filename queue
        imagepath_input = tf.placeholder(tf.string, shape=[self.filename_feed_size])
        target_input = tf.placeholder(tf.float32, shape=[self.filename_feed_size])

        self.filename_queue = tf.FIFOQueue(self.filename_queue_capacity, [tf.string, tf.float32],
                                           shapes=[[], []])
        enqueue_op = self.filename_queue.enqueue_many([imagepath_input, target_input])
        single_path, single_target = self.filename_queue.dequeue()

        # one hot encode the target
        single_target = tf.cast(tf.subtract(single_target, tf.constant(1.0)), tf.int32)
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

        if shuffle:
            images_batch, target_batch = tf.train.shuffle_batch([single_image, single_target],
                                                                batch_size=self.batch_size,
                                                                capacity=self.batch_queue_capacity,
                                                                min_after_dequeue=self.min_after_dequeue,
                                                                num_threads=self.NUM_THREADS)
        else:
            images_batch, target_batch = tf.train.batch([single_image, single_target],
                                                        batch_size=self.batch_size,
                                                        capacity=self.batch_queue_capacity,
                                                        num_threads=1)

        def enqueue(sess):
            under = 0
            max = len(all_img_paths)
            while not self.coord.should_stop():
                upper = under + self.filename_feed_size
                if upper <= max:
                    curr_data = all_img_paths[under:upper]
                    curr_target = all_targets[under:upper]
                    under = upper
                else:
                    rest = upper - max
                    curr_data = np.concatenate((all_img_paths[under:max], all_img_paths[0:rest]))
                    curr_target = np.concatenate((all_targets[under:max], all_targets[0:rest]))
                    under = rest

                sess.run(enqueue_op, feed_dict={imagepath_input: curr_data,
                                                target_input: curr_target})

        enqueue_thread = threading.Thread(target=enqueue, args=[self.sess])

        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)

        enqueue_thread.isDaemon()
        enqueue_thread.start()

        return images_batch, target_batch

    def close(self):
        pass




sess = tf.Session()

calData = CaltechData(100,sess)
print(calData.build_train_data_tensor())





