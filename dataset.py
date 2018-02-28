import pandas as pd
import numpy as np


def import_from_csv(route, class_name = "class", dict = None):
    '''Imports a dataset from a .csv and then converts it into a numpy.ndarray. 
    The class_name specifies the name of the column where the class label is 
    located, which by default is "class".
    It also maps the corresponding tags with the dictionary provided, if any
    '''
    data = pd.read_csv(route)
    data = muestreo_estratificado(data,len(data),class_name)
    if (dict!=None):
        keys_list = list(dict.keys())
        for key in keys_list:
            #print(dict[key])
            data[key] = data[key].map(dict[key])
    return data.as_matrix()

def shuffle_data(data):
    '''Shuffles the dataset randomly	
    '''
    np.random.shuffle(data)

def divide_dataset(data, size):
    '''Divides the dataset into two parts, where the first one has the 
    specified size. It does not change the order of the data	
    '''
    first_batch = data[:size]
    second_batch = data[size:]
    return first_batch, second_batch
    
def get_batch(data, start, size):
    '''It gets a batch from the data, with the starting point and size
    that are provided
    '''
    end = start + size
    batch = data[start:end]
    return batch

def divide_dataset_multiple(data, n_batches):
    '''Divides the dataset into the number of batches that is specified.
    The batches have equal size and if it isn't possible to do an exact
    division then an approximation is used and some data may end up not
    being selected.
    It returns a list with the batches.
    '''
    size = len(data)//n_batches
    batches = []
    for i in range(n_batches):
        batches.append(get_batch(data,i*size,(i+1)*size))
    return batches

def prepare_data_from_csv(route, size, dict = None):
    '''Utility function that imports a dataset from a .csv, then shuffles 
    it and finally divides it in training and test dataset
    '''
    data_array = import_from_csv(route, dict = dict)
    shuffle_data(data_array)
    training, testing = divide_dataset(data_array,size)
    return training, testing

def divide_x_and_y(data, position):
    '''Separates an array into 2 parts: one for the Xs (the income) and 
    the other for the respective Ys (the expected outputs)
    The position is the first index where the outputs are located 
    (e.g. if we have [x,x,x,y,y] with a position=3 we get [x,x,x] and [y,y])	
    '''
    array_x, array_y = np.hsplit(data,[position])
    return array_x, array_y

def get_dict_identity(number):
    '''Utility for the creation of a dictionary that contains the Identity Matrix
    '''
    dictIdentity ={}
    identity = np.identity(number)
    for i in range(number):
        dictIdentity[i] = identity[i]
    return dictIdentity;

#TODO Corregir el muestreo estratificado    
    
# Realiza  un muestreo estratificado
# Recibe:  dataframe(data), el tamaño de la muestra, la columna donde se encuentra la clase (keyClasColumn)
# Retorna un dataframe ordenado por clase
def muestreo_estratificado(datos,size,keyClassColumn):
    clases = datos[keyClassColumn].unique();
    sizeXClass = int(size/len(clases));
    frames = []
    for clase in clases:
        frames.append(datos.loc[datos[keyClassColumn] == clase].head(sizeXClass ))
    return pd.concat(frames)






