import random
from past import dataset as dt, train_conf_past as conf, utils_past as ut, nnet as nn
import numpy as np
import core as co
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from optparse import OptionParser

def obtener_datasetMNIST():
    mnist = input_data.read_data_sets("../../datasets/MNIST_data/", one_hot=True)
    return mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

def obtener_datasetLetterRe(size):
    dict = {}
    dict = ut.add_element_to_dict(dict, 'class', dt.get_dict_identity(26))
    train, test = dt.prepare_data_from_csv("datasets/letter-recognition.csv", size, dict)
    trainX, trainY = dt.divide_x_and_y(train, 16)
    x_test, y_test = dt.divide_x_and_y(test, 16)
    trainY = np.squeeze(trainY)
    y_test = np.squeeze(y_test)

    # El stack se hace para que no quede como array de arrays, sino como matriz (o un array multidimensional)
    trainY = np.stack(trainY)
    y_test = np.stack(y_test)

    return trainX, trainY, x_test, y_test

def obtener_datasetSatelite(size):
    identity = np.identity(7)
    dict = {}
    dict = ut.add_element_to_dict(dict, 'clase', dt.get_dict_identity(7))
    ##paso completamente el archivo a entrenamiento
    train, _ = dt.prepare_data_from_csv("datasets/sattrain.csv", size, dict)
    ##paso completamente el archivo a entrenamiento
    _, test = dt.prepare_data_from_csv("datasets/sattest.csv", 0, dict)
    trainX, trainY = dt.divide_x_and_y(train, 36)
    x_test, y_test = dt.divide_x_and_y(test, 36)
    trainY = np.squeeze(trainY)
    y_test = np.squeeze(y_test)
    # El stack se hace para que no que como array de arrays, sino como matriz (o un array multidimensional)
    y_test = np.stack(y_test)
    trainY = np.stack(trainY)
    return trainX, trainY, x_test, y_test

# Aux. function for configuring parsing options
def config_parser():
    parser = OptionParser()
    parser.add_option("-r", action="store_false", dest="rep", default=True,
                      help="Doesn't execute the version of incremental learning with frontiers")
    parser.add_option("-b", action="store_false", dest="base", default=True,
                      help="Doesn't execute the baseline version of incremental learning")
    parser.add_option("-a", action="store_false", dest="all", default=True,
                      help="Doesn't execute the version with the whole batch")
    parser.add_option("-i", "--iterations", type="int", dest="iterations", default=5, help="Number of tests to be done")
    return parser

# Aux. function for knowing if a test must be skipped, according to the parameters (options) given by the user
def skip_test(mode, options):
    if not options.rep and mode == REPRESENTANTES:
        return True
    elif not options.base and mode == INCREMENTAL_BASE:
        return True
    elif not options.all and mode == COMPLETO:
        return True
    return False


# MODOS DE EJECUCIÓN
REPRESENTANTES = 0  # Indica que se están extrayendo representantes
INCREMENTAL_BASE = 1  # Indica que se está haciendo aprendizaje incremental sin representantes
COMPLETO = 2  # Indica que se pasa todo el dataset una sola vez; no es incremental

# Opciones parser
parser = config_parser()
(options, args) = parser.parse_args()
archivo = open("resultados.txt", 'w')
archivo.write("type;seed;first_batch;second_batch\n")

#trainX, trainY, x_test, y_test = obtener_datasetSatelite(4435)
trainX, trainY, x_test, y_test = obtener_datasetLetterRe(15000)

# ------------VALORES-------------
n_output = trainY.shape[1]
input_layer = trainX.shape[1]

size_batch1 = 10000
n_extra_batches = 5 #Número de lotes incrementales
n_hidden = 70
n_hidden_2 = 50
seeds = []
frontera_x = []
frontera_y = []
centros_x = []
centros_y = []
# ---------------------------------
trainX_B1, trainX_B2 = dt.divide_dataset(trainX, size_batch1)
trainY_B1, trainY_B2 = dt.divide_dataset(trainY, size_batch1)

x_batches = dt.divide_dataset_multiple(trainX_B2,n_extra_batches)
y_batches = dt.divide_dataset_multiple(trainY_B2,n_extra_batches)

print("Size first batch : ", trainX_B1.shape)
print("Size second batch : ", trainX_B2.shape)

for j in range(options.iterations):
    seeds = np.append(seeds, random.randint(1, 100))

for k in range(3):
    # Se salta la prueba si corresponde con las opciones
    if skip_test(k, options):
        continue

    if (k == COMPLETO):
        trainX_B1 = np.concatenate((trainX_B1, trainX_B2), axis=0)
        trainY_B1 = np.concatenate((trainY_B1, trainY_B2), axis=0)

    for iteration in range(options.iterations):
        #-----------------------------------------------------
        #TODO Introducir esta parte del código en una función
        tf.reset_default_graph()
        tf.set_random_seed(seeds[iteration])  # fijamos un valor para la semilla de numeros aleatorios de tensor
        x = tf.placeholder(tf.float32, [None, input_layer])  # representa la entrada
        y = nn.create_neural_net(x,[input_layer,n_hidden,n_hidden_2,n_output], act=tf.nn.tanh)
        configuration = conf.TrainingConfiguration(tf, x, y, x_test, y_test)
        archivo.write(str(k) + ";" + str(seeds[iteration]) + ';')
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        #------------------------------------------------------
        #Se entrena el primer lote
        test_results = configuration.run_basic_test(tf, sess, 500, trainX_B1, trainY_B1)
        archivo.write(str(test_results) + ';')
        if (k == COMPLETO):
            archivo.write('\n')
            continue
        delta_f = 0.2
        delta_c = 0.55
        if (k == REPRESENTANTES):
                _, frontera_x, frontera_y = co.get_fronteras(sess, configuration.x, configuration.y, configuration.y_, trainX_B1, trainY_B1, delta_f, 100)
                _, centros_x, centros_y = co.get_centros(sess, configuration.x, configuration.y, configuration.y_, trainX_B1, trainY_B1, delta_c, 10)
        
        for b in range(len(x_batches)):
            print("\n ::::::ENTRENANDO INCREMENTALMENTE :::::::\n")
            if (k == REPRESENTANTES):
                test_results = configuration.run_prototypes_test(tf, sess, 500, x_batches[b], y_batches[b], frontera_x, frontera_y, centros_x, centros_y)
            else:
                test_results = configuration.run_basic_test(tf, sess, 500, x_batches[b], y_batches[b])
            archivo.write(str(test_results) + ';')
        archivo.write('\n')
        sess.close()

archivo.close()
