import tensorflow as tf
import random
import os

from input.reader import Reader

path = "101_ObjectCategories"

ext_validas = [".jpg", ".gif", ".png", ".jpeg"]


def _find_image_files(path,categories):
	filenames = []
	labels = []
	# LOAD ALL IMAGES 
	for i, category in enumerate(categories):
		iter = 0
		print ("LOAD CATEGORY",category)
		for f in os.listdir(path + "/" + category):
			if iter == 0:
				ext = os.path.splitext(f)[1]
				if ext.lower() not in ext_validas:
					continue
				fullpath = os.path.join(path + "/" + category, f)
				filenames.append(fullpath) # NORMALIZE IMAGE 
				label_curr = i
				labels.append(label_curr)
			#iter = (iter+1)%10;
	shuffled_index = list(range(len(filenames)))
	random.seed(12345)
	random.shuffle(shuffled_index)
	filenames = [filenames[i] for i in shuffled_index]
	labels = [labels[i] for i in shuffled_index]

	print ("Numero filenames: %d" % (len(filenames)))
	print ("Numero labels: %d" % (len(labels)) )
	ncategories =len(categories)
	print (ncategories)

	return filenames,labels


class CaltechReader(Reader):
	data = None
	def __init__(self):
		#Que pasa si no tiene test validation
		super().__init__(path, [path])
		self.categories = sorted(os.listdir(path))
		self.val_filenames, self.val_labels = _find_image_files(path, self.categories)
		self.train_filenames, self.train_labels = _find_image_files(path, self.categories)
					


	def load_class_names(self):
		return self.categories
		
	def load_training_data(self):
		return self.train_filenames, self.train_labels

	def load_test_data(self):
		return self.val_filenames, self.val_labels

	def change_dataset_part(self, index: int):
	    """
	    It changes the target archive of directory from which the training data is being extracted. This ONLY applies
	    to the training data and NOT to the test data.
	    :param index: the number of the mega-batch, starting from 0. I.e. for the first batch, this would be 0
	    :return: None
	    """
	    self.curr_path = self.tr_paths[index]


	@classmethod
	def get_data(cls):
		"""
		Gets the data of Imagenet
		:return: a Singleton object of ImagenetReader
		"""
		if not cls.data:
			cls.data = CaltechReader()
		return cls.data
