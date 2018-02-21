import pandas as pd
import numpy as np

#Imports a dataset from a .csv and then converts it into a numpy.ndarray. It also maps the corresponding tags with the dictionary provided, if any
def import_from_csv(route, dict = None):
    data = pd.read_csv(route)
    if (dict!=None):
        keys_list = list(dict.keys())
        for key in keys_list:
            #print(dict[key])
            data[key] = data[key].map(dict[key])
    return data.as_matrix()

#Shuffles the dataset randomly	
def shuffle_data(data):
    np.random.shuffle(data)

#Divides the dataset into two parts, where the first one has the specified size. It does not change the order of the data	
def divide_dataset(data, size):
    first_batch = data[:size]
    second_batch = data[size:]
    return first_batch, second_batch

#Utility function that imports a dataset from a .csv, then shuffles it and finally divides it in training and test dataset	
def prepare_data_from_csv(route, size, dict = None):
    data_array = import_from_csv(route, dict)
    shuffle_data(data_array)
    training, testing = divide_dataset(data_array,size)
    return training, testing

#Separates an array into 2 parts: one for the Xs (the income) and the other for the respective Ys (the expected outputs)
#The position is the first index where the outputs are located (e.g. if we have [x,x,x,y,y] with a position=3 we get [x,x,x] and [y,y])	
def divide_x_and_y(data, position):
    array_x, array_y = np.hsplit(data,[position])
    return array_x, array_y



##utilidad para crear un diccionario que contiene la matriz idenetidad
def getDictIdentity(number):
    dictIdentity ={}
    identity = np.identity(number)
    for i in range(number):
        dictIdentity[i] = identity[i]
    return dictIdentity;