###############################################################################
# Author:       Imanol Schlag (more info on ischlag.github.io)
# Description:  Functions for loading the imagenet image paths and labels into memory.
# Date:         11.2016
#
#  In order to download the imagenet data you need to look at
#  utils/imagenet_download/run_me.sh
#

import tensorflow as tf
import random
import os

from input.reader import Reader


###############################################################################
# Some TensorFlow Inception functions (ported to python3)
# source: https://github.com/tensorflow/models/blob/master/inception/inception/data/build_imagenet_data.py

def _find_image_files(data_dir, labels_file):
    """Build a list of all images files and labels in the data set.
    Args:
      data_dir: string, path to the root directory of images.
        Assumes that the ImageNet data set resides in JPEG files located in
        the following directory structure.
          data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
          data_dir/n01440764/ILSVRC2012_val_00000543.JPEG
        where 'n01440764' is the unique synset label associated with these images.
      labels_file: string, path to the labels file.
        The list of valid labels are held in this file. Assumes that the file
        contains entries as such:
          n01440764
          n01443537
          n01484850
        where each line corresponds to a label expressed as a synset. We map
        each synset contained in the file to an integer (based on the alphabetical
        ordering) starting with the integer 1 corresponding to the synset
        contained in the first line.
        The reason we start the integer labels at 1 is to reserve label 0 as an
        unused background class.
    Returns:
      filenames: list of strings; each string is a path to an image file.
      synsets: list of strings; each string is a unique WordNet ID.
      labels: list of integer; each integer identifies the ground truth.
    """
    print('Determining list of input files and labels from %s.' % data_dir)
    challenge_synsets = [l.strip() for l in
                         tf.gfile.FastGFile(labels_file, 'r').readlines()]

    labels = []
    filenames = []
    synsets = []

    # Leave label index 0 empty as a background class.
    label_index = 1

    # Construct the list of JPEG files and labels.
    for synset in challenge_synsets:
        jpeg_file_path = '%s/%s/*.JPEG' % (data_dir, synset)
        matching_files = tf.gfile.Glob(jpeg_file_path)

        labels.extend([label_index] * len(matching_files))
        synsets.extend([synset] * len(matching_files))
        filenames.extend(matching_files)

        if not label_index % 100:
            print('Finished finding files in %d of %d classes.' % (
                label_index, len(challenge_synsets)))
        label_index += 1

    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(filenames)))
    random.seed(12345)
    random.shuffle(shuffled_index)

    filenames = [filenames[i] for i in shuffled_index]
    synsets = [synsets[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    print('Found %d JPEG files across %d labels inside %s.' %
          (len(filenames), len(challenge_synsets), data_dir))
    return filenames, synsets, labels


def _find_human_readable_labels(synsets, synset_to_human):
    """Build a list of human-readable labels.
    Args:
      synsets: list of strings; each string is a unique WordNet ID.
      synset_to_human: dict of synset to human labels, e.g.,
        'n02119022' --> 'red fox, Vulpes vulpes'
    Returns:
      List of human-readable strings corresponding to each synset.
    """
    humans = []
    for s in synsets:
        assert s in synset_to_human, ('Failed to find: %s' % s)
        humans.append(synset_to_human[s])
    return humans


def _build_synset_lookup(imagenet_metadata_file):
    """Build lookup for synset to human-readable label.
    Args:
      imagenet_metadata_file: string, path to file containing mapping from
        synset to human-readable label.
        Assumes each line of the file looks like:
          n02119247    black fox
          n02119359    silver fox
          n02119477    red fox, Vulpes fulva
        where each line corresponds to a unique mapping. Note that each line is
        formatted as <synset>\t<human readable label>.
    Returns:
      Dictionary of synset to human labels, such as:
        'n02119022' --> 'red fox, Vulpes vulpes'
    """
    lines = tf.gfile.FastGFile(imagenet_metadata_file, 'r').readlines()
    synset_to_human = {}
    for l in lines:
        if l:
            parts = l.strip().split('\t')
            assert len(parts) == 2
            synset = parts[0]
            human = parts[1]
            synset_to_human[synset] = human
    return synset_to_human


###############################################################################

class ImagenetReader(Reader):
    """
    Reader for Tiny Imagenet dataset
    """
    __train_dirs, __validation_dir, __extras = None, None, None
    data = None

    def __init__(self, train_dirs: [str], validation_dir: str, extras: [str]):
        """
        Creates an ImagenetReader object
        :param train_dirs: the paths to the training data
        :param validation_dir: the path to the testing data
        :param extras: an array with extra paths, must be of this form:
                        [labels_file_path, metadata_file_path]
        """
        super().__init__(train_dirs, validation_dir)
        self.synset_to_human = _build_synset_lookup(extras[1])
        self.labels_file = extras[0]

        self.val_filenames, self.val_synsets, self.val_labels = _find_image_files(self.test_path, self.labels_file)
        self.train_filenames, self.train_synsets, self.train_labels = _find_image_files(self.curr_path,
                                                                                        self.labels_file)
        self.humans = _find_human_readable_labels(self.val_synsets, self.synset_to_human)

    def load_class_names(self):
        return self.humans

    def load_training_data(self):
        return self.train_filenames, self.train_labels

    def load_test_data(self):
        return self.val_filenames, self.val_labels

    @classmethod
    def get_data(cls):
        """
        Gets the data of Imagenet. set_parameters must be called before this method or an Exception may be raised.
        :return: a Singleton object of ImagenetReader
        """
        if not cls.data:
            cls.data = ImagenetReader(cls.__train_dirs, cls.__validation_dir, cls.__extras)
        return cls.data

    @classmethod
    def set_parameters(cls, train_dirs: [str], validation_dir: str, extras: [str]):
        """
        Sets the parameters for the Singleton reader
        :param train_dirs: the paths to the training data
        :param validation_dir: the path to the testing data
        :param extras: an array with extra paths, must be of this form:
                        [labels_file_path, metadata_file_path]
        """
        cls.__train_dirs = train_dirs
        cls.__validation_dir = validation_dir
        cls.__extras = extras

    def reload_training_data(self):
        self.train_filenames, self.train_synsets, self.train_labels = _find_image_files(self.curr_path,
                                                                                        self.labels_file)
