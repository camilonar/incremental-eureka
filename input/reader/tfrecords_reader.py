"""
Module for reading a dataset stored in TFRecords
"""
from input.reader import Reader


class TFRecordsReader(Reader):
    """
    Reader for datasets stored in TFRecords. Its only purpouse is to store the training and testing data paths
    """

    def reload_training_data(self):
        pass

    def __init__(self, train_dirs: [str], validation_dir: str):
        super().__init__(train_dirs, validation_dir)
        print("TEST PATH ", validation_dir)
        print("TRAIN PATHS ", train_dirs)

    def load_training_data(self):
        return self.curr_path, None

    def load_test_data(self):
        return self.test_path, None
