
import tensorflow as tf
import random
import os
import pickle
import numpy as np

from input.reader import Reader

size_image = 32
numero_canales = 3

base_folder="../datasets/cifar-10-batches-py"

base = base_folder+"/data_batch_"
tr_paths = [base+"1",base+"2",base+"3",base+"4",base+"5"]
test_path = base_folder+"/test_batch"
metadata_file = base_folder+"/batches.meta"


number_of_class = 10

def _convert_raw_to_image(raw):
	"""
	Convierte las imagenes desde el formato cifar 10 
	y retorna un arreglo de 4 dimensiones [numero de imagenes , alto , ancho , numero de canales]
	 cada pixel se representa por un numero flotante de 0 a 1 
	"""
	# Convierte las imgenes sin procesar a valores numericos 
	raw_float = np.array(raw, dtype=float) / 255.0

	# reshape
	images = raw_float.reshape([-1, numero_canales, size_image, size_image])

	# reordena los indices de el array 
	images = images.transpose([0, 2, 3, 1])

	return images


def _to_one_hot(class_numbers, num_classes=None):
	if num_classes is None:
		num_classes = np.max(class_numbers) + 1

	return np.eye(num_classes, dtype=float)[class_numbers]


def _unpickle(filename):

	print("Loading data: " + filename)
	with open(filename, mode='rb') as file:
		data = pickle.load(file, encoding='bytes')
	return data

def _get_human_readable_labels():
	raw = _unpickle(filename=metadata_file)[b'label_names']
	humans = [x.decode('utf-8') for x in raw]
	return humans

def _load_batch(filename):
	"""
	load file serialized of dataset cifar10 
	"""
	# Carga el archivo serializado 
	data = _unpickle(filename)
	#obtaing images in format raw *(serialized)
	raw_image = data[b'data']
	#obtain number of class 
	cls = np.array(data[b'labels'])
	images = _convert_raw_to_image(raw_image)
	return images, cls


def _load_data(filename):
	images, cls = _load_batch(filename)
	return images, cls, _to_one_hot(class_numbers=cls, num_classes=number_of_class)




class CifarReader(Reader):

	def __init__(self):
		super().__init__(test_path, tr_paths)
		print("TEST PATH ",test_path)
		print("TRAIN PATHs ",tr_paths)        	  	

	def load_class_names(self):
		return _get_human_readable_labels()

	def load_training_data(self):

		return _load_data(self.curr_path)

	def load_test_data(self):
		return _load_data(self.test_path)

	def reload_training_data(self):
		pass

	@classmethod
	def get_data(cls):
		"""
		Gets the data of Imagenet
		:return: a Singleton object of CifarReader
		"""
		if not cls.data:
			cls.data = CifarReader()
		return cls.data

