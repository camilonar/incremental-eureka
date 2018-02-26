import pandas as pd
import numpy as np

#Imports a dataset from a .csv and then converts it into a numpy.ndarray. 
#The class_name specifies the name of the column where the class label is located, which by default is "class".
#It also maps the corresponding tags with the dictionary provided, if any
def import_from_csv(route, class_name = "class", dict = None):
    data = pd.read_csv(route)
    data = muestreo_estratificado(data,len(data),class_name)
    print(len(data))
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
    data_array = import_from_csv(route, dict = dict)
    shuffle_data(data_array)
    training, testing = divide_dataset(data_array,size)
    return training, testing

#Separates an array into 2 parts: one for the Xs (the income) and the other for the respective Ys (the expected outputs)
#The position is the first index where the outputs are located (e.g. if we have [x,x,x,y,y] with a position=3 we get [x,x,x] and [y,y])	
def divide_x_and_y(data, position):
    array_x, array_y = np.hsplit(data,[position])
    return array_x, array_y



#Utilidad para crear un diccionario que contiene la matriz identidad
def getDictIdentity(number):
    dictIdentity ={}
    identity = np.identity(number)
    for i in range(number):
        dictIdentity[i] = identity[i]
    return dictIdentity;

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






