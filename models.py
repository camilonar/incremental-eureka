"""
Module containing various useful neural networks models
"""

import tensorflow as tf
from tensorflow.contrib.layers import flatten
from network import Network

# TODO: revisar las funciones de activación
# TODO: agregar líneas necesarias para los summaries
from network import Network


def create_LeNet(input):
    """
    Creates a Neural Net with LeNet-5 architecture. This code was taken and adapted from:
    https://github.com/sujaybabruwad/LeNet-in-Tensorflow
    :param input: the input tensor of the network. It must follow the shape [None,32,32,C], where C is the number
    of color channels of the images, and C>=1. If the images are grayscale then C=1.
    :return: the final layer (output layer) o a CNN with LeNet architecture
    """
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1': 6,
        'layer_2': 20,
        'layer_3': 120,
        'layer_f1': 84,
        'outputs': 10,
    }

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, layer_depth['layer_1']], mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(layer_depth['layer_1']))
    conv1 = tf.nn.conv2d(input, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(
        tf.truncated_normal(shape=[5, 5, layer_depth['layer_1'], layer_depth['layer_2']], mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(layer_depth['layer_2']))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(
        tf.truncated_normal(shape=(5 * 5 * layer_depth['layer_2'], layer_depth['layer_3']), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(layer_depth['layer_3']))
    fc1 = tf.matmul(fc1, fc1_w) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(
        tf.truncated_normal(shape=(layer_depth['layer_3'], layer_depth['layer_f1']), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(layer_depth['layer_f1']))
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b
    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(
        tf.truncated_normal(shape=(layer_depth['layer_f1'], layer_depth['outputs']), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(layer_depth['outputs']))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits


# TODO: revisar las funciones de activación
# TODO: agregar líneas necesarias para los summaries
# TODO: modificar la red para adaptarla a los valores de los artículos
class CaffeNet(Network):
    def setup(self):
        """
        Creates a Neural Net with a simplified CaffeNet architecture. The input data must have been previously set in
         the constructor of the object as 'data'.
         E.g.:
            net = CaffeNet({'data': input_tensor})

         This code was taken and adapted from:
            https://github.com/ethereon/caffe-tensorflow
        :return: an instance of CaffeNet
        """
        (self.feed('data')
         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
         .lrn(2, 2e-05, 0.75, name='norm1')
         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .lrn(2, 2e-05, 0.75, name='norm2')
         .conv(3, 3, 384, 1, 1, name='conv3')
         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
         .fc(4096, name='fc6')
         .fc(4096, name='fc7')
         .fc(1000, relu=False, name='fc8')
         .softmax(name='prob'))
