
import tensorflow as tf
import numpy as np
import threading

import cifar_reader as cifar
from input.data import Data

class CifarData(Data):
	NUM_THREADS = 8
	NUMBER_OF_CLASSES = 101
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
		print("Loading Cifar10 data...")
		my_cifar = cifar.CifarReader()
		super().__init__(batch_size, sess, my_cifar, image_height, image_width)
		#self.filename_feed_size = filename_feed_size
		#self.filename_queue_capacity = filename_queue_capacity
		#self.batch_queue_capacity = batch_queue_capacity + 3 * batch_size
		#self.min_after_dequeue = min_after_dequeue
		self.data_reader.check_if_downloaded()

	def build_train_data_tensor(self, shuffle=False, augmentation=False):
		
		pass
	def build_test_data_tensor(self, shuffle=False, augmentation=False):

		pass
	def change_dataset_part(self, index: int):

		pass
	def __del__(self):
		
		pass
	def close(self):
		"""
		Closes the pipeline
		:return: None
		"""
		raise NotImplementedError("The subclass hasn't implemented the close method")