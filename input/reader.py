"""
This module is used to abstract the reading of the data from disk
"""
from typing import List
import os


class Reader(object):
    """Interface for the reading of data (of a dataset) from disk.

    This structure is based in the pipelines from:
        https://github.com/ischlag/tensorflow-input-pipelines"""

    def __init__(self, test_path: str, tr_paths: List[str]):
        """
        Creates a Reader object and points the current training data path to the first one of the list provided
        :param test_path: the path where the test/validation data is located
        :param tr_paths: a list of paths, where each one corresponds with the location of one part of the dataset,
        which means, that it has the location of each one of the mega-batches
        """
        self.test_path = test_path
        self.tr_paths = tr_paths
        self.curr_path = self.tr_paths[0]

    def load_training_data(self):
        """
        It loads and prepares the training data to be fed to the input pipeline
        :return: the filenames of the images and labels of the training data of the current mega-batch
        """
        raise NotImplementedError("The subclass hasn't implemented the load_training_data method")

    def load_test_data(self):
        """
        It loads and prepares the test data to be fed to the input pipeline
        :return: the filenames of the images and labels of the test data of the current mega-batch
        """
        raise NotImplementedError("The subclass hasn't implemented the load_test_data method")

    def check_if_downloaded(self):
        """
        Checks if the directories or files with training and test data exists
        :return: None
        :raises Exception: if the data is not found
        """
        for i, path in enumerate(self.tr_paths):
            if os.path.exists(path):
                print("Train directory for batch {} seems to exist".format(i))
            else:
                raise Exception("Train directory for batch {} doesn't seem to exist".format(i))

        if os.path.exists(self.test_path):
            print("Validation directory seems to exist")
        else:
            raise Exception("Validation directory doesn't seem to exist.")

    def change_dataset_part(self, index: int):
        """
        It changes the target archive of directory from which the training data is being extracted. This ONLY applies
        to the training data and NOT to the test data.
        :param index: the number of the mega-batch, starting from 0. I.e. for the first batch, this would be 0
        :return: None
        """
        self.curr_path = self.tr_paths[index]
