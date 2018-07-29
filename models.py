"""
Module containing various useful neural networks models
"""

import tensorflow as tf
from tensorflow.contrib.layers import flatten
from network import Network


# TODO: revisar las funciones de activación
# TODO: agregar líneas necesarias para los summaries
class LeNet(Network):
    def setup(self):
        """
        Creates a Neural Net with LeNet-5 architecture. The input shape of the tensor must follow the shape
        [None,32,32,C], where C is the number of color channels of the images, and C>=1. If the images are grayscale
        then C=1. This Tensor must have been previously set in the constructor of the object as 'data'
        E.g.:
            net = LeNet({'data': input_tensor})

        This code was taken and adapted from:
        https://github.com/sujaybabruwad/LeNet-in-Tensorflow
        """
        (self.feed('data')
         .conv(5, 5, 6, 1, 1, padding='VALID', name='conv1')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(5, 5, 20, 1, 1, padding='VALID', name='conv2')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .fc(120, name='fc1')
         .fc(84, name='fc2')
         .fc(10, relu=False, name='fc3'))


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
        :return: None
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
