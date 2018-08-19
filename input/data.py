"""
Module for data input.
"""
from abc import abstractmethod
from input.reader import Reader
from training.train_conf import GeneralConfig


class Data(object):
    """
    This class acts as an interface for data pipelines.

    This structure is based in the pipelines from:
        https://github.com/ischlag/tensorflow-input-pipelines
    """

    def __init__(self, general_config: GeneralConfig, data_reader: Reader,
                 image_height, image_width):
        """
        Creates a Data pipeline object for a dataset composed of images. It also sets the current configuration for
        training as the configuration for the first mega-batch.

        This must be called by the constructors of the subclasses.
        :param general_config: the configuration for the whole training
        :param data_reader: the corresponding Reader of the data of the dataset
        :param image_height: the height at which the images are going to be rescaled
        :param image_width: the width at which the images are going to be rescaled
        """
        self.general_config = general_config
        self.curr_config = self.general_config.train_configurations[0]
        self.data_reader = data_reader
        self.image_height = image_height
        self.image_width = image_width

    @abstractmethod
    def build_train_data_tensor(self, shuffle=False, augmentation=False, skip_count=0):
        """
        Builds the training data tensor
        :param shuffle: specifies whether the data is going to be randomly shuffled or not
        :param augmentation: specifies if there is going to be performed a data augmentation process
        :param skip_count: number of elements to be skipped from the Dataset. If the dataset.batch is applied, then each
        batch is treated as 1 element, so in that case, skip_count is the number of batches to be skipped from the
        Dataset.
        :return: a tuple of Tensors, where the first value corresponds with the tensor of training data and the second
        is the tensor with the corresponding labels of the data
        """
        raise NotImplementedError("The subclass hasn't implemented the build_train_data_tensor method")

    @abstractmethod
    def build_test_data_tensor(self, shuffle=False, augmentation=False):
        """
        Builds the test data tensor
        :param shuffle: specifies whether the data is going to be randomly shuffled or not
        :param augmentation: specifies if there is going to be performed a data augmentation process
        :return: a tuple of Tensors, where the first value corresponds with the tensor of test data and the second is
        the tensor with the corresponding labels of the data
        """
        raise NotImplementedError("The subclass hasn't implemented the build_test_data_tensor method")

    def change_dataset_part(self, index: int):
        """
        It changes the target archive of directory from which the training data is being extracted. This ONLY applies
        to the training data and NOT to the test data.
        :param index: the number of the mega-batch, starting from 0. I.e. for the first batch, this would be 0
        :return: None
        """
        print("Changing dataset part to part {} in the Data object...".format(index))
        self.data_reader.change_dataset_part(index)
        self.curr_config = self.general_config.train_configurations[index]
        self.close()

    def __del__(self):
        """
        Destroys the object and closes the pipeline
        :return: None
        """
        self.close()

    @abstractmethod
    def close(self):
        """
        Closes the pipeline
        :return: None
        """
        raise NotImplementedError("The subclass hasn't implemented the close method")
