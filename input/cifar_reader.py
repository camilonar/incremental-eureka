import pickle
import numpy as np

from input.reader import Reader

size_image = 32
numero_canales = 3

number_of_classes = 10


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
    raw = _unpickle(filename=CifarReader._metadata_file)[b'label_names']
    humans = [x.decode('utf-8') for x in raw]
    return humans


def _load_batch(filename):
    """
    load file serialized of dataset CIFAR-10
    """
    # Carga el archivo serializado
    data = _unpickle(filename)
    # obtaing images in format raw *(serialized)
    raw_image = data[b'data']
    # obtain number of class
    cls = np.array(data[b'labels'])
    images = _convert_raw_to_image(raw_image)
    return images, cls


def _load_data(filename):
    images, cls = _load_batch(filename)
    return images, cls, _to_one_hot(class_numbers=cls, num_classes=number_of_classes)


class CifarReader(Reader):

    __train_dirs, __validation_dir, _metadata_file = None, None, None
    data = None

    def reload_training_data(self):
        self.imgs_raw, _, self.cls_raw = _load_data(self.curr_path)

    def __init__(self, train_dirs: [str], validation_dir: str):
        super().__init__(train_dirs, validation_dir)
        print("TEST PATH ", validation_dir)
        print("TRAIN PATHs ", train_dirs)
        self.imgs_raw, _, self.cls_raw = _load_data(self.curr_path)

    def load_class_names(self):
        return _get_human_readable_labels()

    def load_training_data(self):
        return self.imgs_raw, self.cls_raw

    def load_test_data(self):
        return _load_data(self.test_path)

    @classmethod
    def get_data(cls):
        """
        Gets the data of CIFAR-10. set_parameters must be called before this method or an Exception may be raised.
        :return: a Singleton object of CifarReader
        """
        if not cls.data:
            cls.data = CifarReader(cls.__train_dirs, cls.__validation_dir)
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
        cls._metadata_file = extras[0]
