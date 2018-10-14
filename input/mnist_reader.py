"""
Module for reading the MNIST dataset.
"""
from input.reader import Reader


class MnistReader(Reader):
    """
    Reader for MNIST dataset
    """
    __train_dirs, __validation_dir, _metadata_file = None, None, None
    data = None

    def __init__(self, train_dirs: [str], validation_dir: str):
        super().__init__(train_dirs, validation_dir)
        print("TEST PATH ", validation_dir)
        print("TRAIN PATHS ", train_dirs)

    def load_training_data(self):
        # return self.curr_path, None
        return self.tr_paths, None

    def load_test_data(self):
        return self.test_path, None

    def load_class_names(self):
        pass

    def reload_training_data(self):
        pass

    @classmethod
    def get_data(cls):
        """
        Gets the data of MNIST
        :return: a Singleton object of MnistReader
        """
        if not cls.data:
            cls.data = MnistReader(cls.__train_dirs, cls.__validation_dir)
        return cls.data

    @classmethod
    def set_parameters(cls, train_dirs: [str], validation_dir: str, extras: [str]):
        """
        Sets the parameters for the Singleton reader
        :param train_dirs: the paths to the training data
        :param validation_dir: the path to the testing data
        :param extras: an array with extra paths, must be of this form:
                        [metadata_file_path]
        """
        cls.__train_dirs = train_dirs
        cls.__validation_dir = validation_dir
