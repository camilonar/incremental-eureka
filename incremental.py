import sys
import random
import dataset as dt
import utils as ut
import numpy as np
import nnet as nn
import core as co
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from optparse import OptionParser


def obtener_datasetMNIST():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    return mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


def obtener_datasetLetterRe():
    dict = {}
    dict = ut.add_element_to_dict(dict, 'class', dt.getDictIdentity(26))
    train, test = dt.prepare_data_from_csv("letter-recognition.csv", 15000, dict)
    trainX, trainY = dt.divide_x_and_y(train, 16)
    x_test, y_test = dt.divide_x_and_y(test, 16)
    trainY = np.squeeze(trainY)
    y_test = np.squeeze(y_test)

    # El stack se hace para que no quede como array de arrays, sino como matriz (o un array multidimensional)
    trainY = np.stack(trainY)
    y_test = np.stack(y_test)

    return trainX, trainY, x_test, y_test


def obtener_datasetSatelite():
    identity = np.identity(7)
    dict = {}
    dict = ut.add_element_to_dict(dict, 'clase', dt.getDictIdentity(7))
    ##paso completamente el archivo a entrenamiento
    train, _ = dt.prepare_data_from_csv("sattrain.csv", 4435, dict)
    ##paso completamente el archivo a entrenamiento
    _, test = dt.prepare_data_from_csv("sattest.csv", 0, dict)
    trainX, trainY = dt.divide_x_and_y(train, 36)
    x_test, y_test = dt.divide_x_and_y(test, 36)
    trainY = np.squeeze(trainY)
    y_test = np.squeeze(y_test)
    # El stack se hace para que no que como array de arrays, sino como matriz (o un array multidimensional)
    y_test = np.stack(y_test)
    trainY = np.stack(trainY)
    return trainX, trainY, x_test, y_test


# Divides a dataset in two parts
def split_dataset(x, y, offset, size):
    return x[offset:size], y[offset:size]


# Test the neural network against the training dataset and a test dataset to review performance
def probar_dataset(sess, x_input_v, y_input_v, x_test_v, y_test_v):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_result = 1.0 - sess.run(accuracy, feed_dict={x: x_test_v, y_: y_test_v})
    train_result = 1.0 - sess.run(accuracy, feed_dict={x: x_input_v, y_: y_input_v})

    print("Resultados Test: " + str(test_result))
    print("Resultados Training: " + str(train_result))

    return test_result, train_result


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

tf.reset_default_graph()
tf.set_random_seed(59)

#trainX, trainY, x_test, y_test = obtener_datasetSatelite()
trainX, trainY, x_test, y_test = obtener_datasetLetterRe()

# ------------VALORES-------------
n_output = trainY.shape[1]
input_layer = trainX.shape[1]

size_batch = 10000
size_second_batch = 5000
n_hidden = 70
n_hidden_2 = 50
seeds = []
frontera_x = []
frontera_y = []
centros_x = []
centros_y = []
# ---------------------------------

trainX_B1, trainY_B1 = split_dataset(trainX, trainY, 0, size_batch)  # obtenemos los datos para el primer batch
trainX_B2, trainY_B2 = split_dataset(trainX, trainY, size_batch,
                                     size_batch + size_second_batch)  # obtenemos los datos para el segundo batch - se obtienen los siguientes 1000 datos

print("Size first batch : ", trainX_B1.shape)
print("Size second batch : ", trainX_B2.shape)

# definimos los pesos  (que tan importante es ) en 1
x_pesosDefault = np.full((size_batch, n_output), 1)

for j in range(options.iterations):
    seeds = np.append(seeds, random.randint(1, 100))

for k in range(3):
    # Se salta la prueba si corresponde con las opciones
    if skip_test(k, options):
        continue

    if (k == COMPLETO):
        trainX_B1 = np.concatenate((trainX_B1, trainX_B2), axis=0)
        trainY_B1 = np.concatenate((trainY_B1, trainY_B2), axis=0)
        x_pesosDefault = np.append(x_pesosDefault, np.full((size_second_batch, n_output), 1), axis=0)

    for iteration in range(options.iterations):
        tf.reset_default_graph()
        tf.set_random_seed(seeds[iteration])  # fijamos un valor para la semilla de numeros aleatorios de tensor
        x_pesos = tf.placeholder(tf.float32, [None, n_output])
        x = tf.placeholder(tf.float32, [None, input_layer])  # representa la entrada
        y_ = tf.placeholder(tf.float32, [None, n_output])  # representa la salida deseada
        y = nn.create_neural_net(x,[input_layer,n_hidden,n_hidden_2,n_output], act=tf.nn.tanh)
        mse = tf.reduce_mean(tf.square(y - y_) * x_pesos)
        train_step = tf.train.AdamOptimizer(0.004).minimize(mse)
        archivo.write(str(k) + ";" + str(seeds[iteration]) + ';')
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)
        for i in range(500):
            _, c = sess.run([train_step, mse], feed_dict={x: trainX_B1, y_: trainY_B1, x_pesos: x_pesosDefault})
            if i % 10 == 0:
                print(i * 100 / 150, " val-> OK")
        print("\n:::PROBANDO:::::::::::\n")
        test_results, _ = probar_dataset(sess, trainX_B1, trainY_B1, x_test, y_test)
        archivo.write(str(test_results) + ';')
        if (k == COMPLETO):
            archivo.write('\n')
            continue
        delta_f = 0.2
        delta_c = 0.55
        x_pesosConRep = np.full((len(trainX_B2), n_output), 1)
        if (k == REPRESENTANTES):
            _, frontera_x, frontera_y = co.get_fronteras(sess, x, y, y_, trainX_B1, trainY_B1, delta_f, 250)
            _, centros_x, centros_y = co.get_centros(sess, x, y, y_, trainX_B1, trainY_B1, delta_c, 20)
            x_pesosConRep = np.append(x_pesosConRep, np.full((len(frontera_x) + len(centros_x), n_output), 3), axis=0)
            trainX_B2_aux = np.concatenate((trainX_B2, np.asanyarray(frontera_x)), axis=0)
            trainX_B2_aux = np.concatenate((trainX_B2_aux, np.asanyarray(centros_x)), axis=0)
            trainY_B2_aux = np.concatenate((trainY_B2, np.asanyarray(frontera_y)), axis=0)
            trainY_B2_aux = np.concatenate((trainY_B2_aux, np.asanyarray(centros_y)), axis=0)
            print("shape trainX ", trainX_B2_aux.shape)
            print("shape trainY ", trainY_B2_aux.shape)
            print("shape fronteraX ", np.asanyarray(frontera_x).shape)
            print("shape fronteraY ", np.asanyarray(frontera_y).shape)
        else:
            trainX_B2_aux = np.copy(trainX_B2)
            trainY_B2_aux = np.copy(trainY_B2)
        print("\n ::::::ENTRENANDO SEGUNDO LOTE CON :::::::\n")

        for i in range(500):  # gradiente descendente estocastico
            _, c = sess.run([train_step, mse], feed_dict={x: trainX_B2_aux, y_: trainY_B2_aux, x_pesos: x_pesosConRep})
            if i % 500 == 0:
                print('OK')
        test_results, _ = probar_dataset(sess, trainX_B2, trainY_B2, x_test, y_test)
        probar_dataset(sess, trainX_B1, trainY_B1, x_test, y_test)
        archivo.write(str(test_results) + '\n')
        sess.close()

archivo.close()
