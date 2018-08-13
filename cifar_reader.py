
import tensorflow as tf
import random
import os
import pickle

from input.reader import Reader
base_folder="cifar-10-batches-py"
base = base_folder+"/data_batch_"
tr_paths = [base+"1",base+"2",base+"3",base+"4",base+"5"]
test_path = base_folder+"/test_batch"
metadata_file = base_folder+"/batches.meta"


def _get_path(filename=""):
    #Return full path  
    return os.path.join(base_folder, filename)

def _unpickle(filename):
    file_path = _get_path(filename)
    print("Loading data: " + file_path)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data

def _get_human_readable_labels():
    raw = _unpickle(filename="batches.meta")[b'label_names']
    humans = [x.decode('utf-8') for x in raw]
    return humans

def _load_current_training_data():
    """
    Carga todo el set de entrenamiento del dataset 
    """
    images = np.zeros(shape=[numero_imagenes_entreamiento, size_image, size_image, numero_canales], dtype=float)
    cls = np.zeros(shape=[numero_imagenes_entreamiento], dtype=int)
    batch_inicio = 0
    for i in range(numero_lotes):
        images_batch, cls_batch = cargar_datos(filename="data_batch_" + str(i + 1))
        num_images = len(images_batch)
        end = batch_inicio + num_images

        # almacena las imagenes dentro del array 
        images[batch_inicio:end, :] = images_batch

        # almacena el numero de clases dentro del array 
        cls[batch_inicio:end] = cls_batch

        batch_inicio = end

    return images, cls, codificar_a_one_hot(class_numbers=cls, num_classes=numero_clases)




class CifarReader(Reader):

    def __init__(self):
    	pass    	  	

    def check_if_downloaded(self):
        if os.path.exists(base_folder):
            print("Train directory seems to exist")
        else:
            raise Exception("Train directory doesn't seem to exist.")

        if os.path.exists(test_path):
            print("Validation directory seems to exist")
        else:
            raise Exception("Validation directory doesn't seem to exist.")



    def load_class_names(self):
    	return _get_human_readable_labels()

    def load_training_data(self):
        
    	return 

    def load_test_data(self):
    	pass


data = CifarReader()

print(data.load_class_names())