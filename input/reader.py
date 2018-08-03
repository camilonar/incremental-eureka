"""
This module is used to abstract the reading of the data from disk
"""


# TODO documentar, agregar argumentos(si se necesita) e implementar
# TODO agregar función de cargue de múltiples partes del dataset
class Reader(object):
    """Interface for the reading of data (of a dataset) from disk.

    This structure is based in the pipelines from:
        https://github.com/ischlag/tensorflow-input-pipelines"""

    def __init__(self):
        pass

    def load_training_data(self):
        pass

    def load_test_data(self):
        pass

    def check_if_downloaded(self):
        pass

    def change_dataset_part(self, index: int):
        """
        It changes the target archive of directory from which the training data is being extracted. This ONLY applies
        to the training data and NOT to the test data.
        :param index: the number of the mega-batch, starting from 0. I.e. for the first batch, this would be 0
        :return: None
        """
        pass
