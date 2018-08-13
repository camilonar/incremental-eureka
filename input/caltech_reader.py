import tensorflow as tf
import random
import os

from input.reader import Reader

# TODO múltiples mega lotes, en vez de sólo 1
path = "../datasets/101_ObjectCategories"

ext_validas = [".jpg", ".gif", ".png", ".jpeg"]

###############################################################################
# Some TensorFlow Inception functions (ported to python3)
# source: https://github.com/tensorflow/models/blob/master/inception/inception/data/build_imagenet_data.py
def _find_image_files(path, categories):
    filenames = []
    labels = []
    # LOAD ALL IMAGES
    for i, category in enumerate(categories):
        iter = 0
        print("LOAD CATEGORY", category)
        for f in os.listdir(path + "/" + category):
            if iter == 0:
                ext = os.path.splitext(f)[1]
                if ext.lower() not in ext_validas:
                    continue
                fullpath = os.path.join(path + "/" + category, f)
                filenames.append(fullpath)  # NORMALIZE IMAGE
                label_curr = i
                labels.append(label_curr)
    # iter = (iter+1)%10;
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)
    filenames = [filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print("Numero filenames: %d" % (len(filenames)))
    print("Numero labels: %d" % (len(labels)))
    ncategories = len(categories)
    print(ncategories)

    return filenames, labels


class CaltechReader(Reader):
    """
    Reader for Caltech101 dataset
    """
    data = None

    def __init__(self):
        # TODO Que pasa si no tiene test validation
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

    @classmethod
    def get_data(cls):
        """
        Gets the data of Imagenet
        :return: a Singleton object of ImagenetReader
        """
        if not cls.data:
            cls.data = CaltechReader()
        return cls.data

    def reload_training_data(self):
        self.train_filenames, self.train_labels = _find_image_files(path, self.categories)
