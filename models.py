"""
Module containing various useful neural networks models
"""
import tensorflow as tf
from keras.applications import xception
from libs.caffe_tensorflow.network import Network
from utils import default_paths as paths


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


class NiN(Network):
    def setup(self):
        """
        Creates a Neural Net with NiN architecture. The input data must have been previously set in
        the constructor of the object as 'data'.
        E.g.:
            net = NiN({'data': input_tensor})
        :return: None
        """
        (self.feed('data')
         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
         .conv(1, 1, 96, 1, 1, name='mlp1')
         .conv(1, 1, 96, 1, 1, name='mlp2')
         .max_pool(3, 3, 2, 2, name='pool1')
         .dropout(0.5, name='drop1')
         .conv(5, 5, 256, 1, 1, name='conv2')
         .conv(1, 1, 256, 1, 1, name='mlp3')
         .conv(1, 1, 256, 1, 1, name='mlp4')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .dropout(0.5, name='drop2')
         .conv(3, 3, 384, 1, 1, name='conv3')
         .conv(1, 1, 384, 1, 1, name='mlp5')
         .conv(1, 1, 384, 1, 1, name='mlp6')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 1024, 1, 1, name='conv4')
         .conv(1, 1, 1024, 1, 1, name='mlp7')
         .conv(1, 1, 100, 1, 1, name='mlp8')
         .avg_pool(6, 6, 1, 1, padding='VALID', name='pool4'))

    def get_output(self):
        return tf.squeeze(self.terminals[-1])


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
         .conv(5, 5, 32, 1, 1, padding='VALID', name='conv1')
         .conv(5, 5, 32, 1, 1, group=2, name='conv2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 64, 1, 1, name='conv3')
         .conv(3, 3, 64, 1, 1, name='conv4')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
         .fc(512, name='fc6')
         .dropout(0.5, name='drop6')
         .fc(12, relu=False, name='fc8'))


class DenseNet(Network):

    def setup(self):
        growth_k = 12
        num_class= 100
        """
            to 256 d 256 images
        :return: None
        """
        (self.feed('data')
         .conv(7, 7, growth_k * 2, 2, 2, name="conv_1")
         .dense_block(nb_layers=6, growth_k=growth_k, dropout_rate=0.2, name="dense_1")
         .transition_layer(growth_k=growth_k,dropout_rate=0.2,name="trans_1")

         .dense_block(nb_layers=12, growth_k=growth_k, dropout_rate=0.2, name="dense_2")
         .transition_layer(growth_k=growth_k, dropout_rate=0.2, name="trans_2")

         .dense_block(nb_layers=48, growth_k=growth_k, dropout_rate=0.2, name="dense_3")
         .transition_layer(growth_k=growth_k, dropout_rate=0.2, name="trans_3")
         .dense_block(nb_layers=32, growth_k=growth_k, dropout_rate=0.2, name="dense_final")

         .batch_normalization(name="linear_batch")
         .relu(name="relu")
         .global_average_pooling()
         .flatten()
         .linear(class_num=num_class, name='linear')
         )


class VGGNet(Network):
    def setup(self):
        """
        Creates a Neural Net with a VGG16 architecture. The input data must have been previously set in
         the constructor of the object as 'data'.
         E.g.:
            net = VGGNet({'data': input_tensor})
        :return: None
        """
        (self.feed('data')
         .conv(3, 3, 64, 1, 1, padding='SAME', name='conv1_1')
         .conv(3, 3, 64, 1, 1, padding='SAME', name='conv1_2')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool1')
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv2_1')
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv2_2')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool2')
         .conv(3, 3, 256, 1, 1, padding='SAME', name='conv3_1')
         .conv(3, 3, 256, 1, 1, padding='SAME', name='conv3_2')
         .conv(3, 3, 256, 1, 1, padding='SAME', name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool3')
         .conv(3, 3, 512, 1, 1, padding='SAME', name='conv4_1')
         .conv(3, 3, 512, 1, 1, padding='SAME', name='conv4_2')
         .conv(3, 3, 512, 1, 1, padding='SAME', name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool4')
         .conv(3, 3, 512, 1, 1, padding='SAME', name='conv5_1')
         .conv(3, 3, 512, 1, 1, padding='SAME', name='conv5_2')
         .conv(3, 3, 512, 1, 1, padding='SAME', name='conv5_3')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool5')
         .fc(256, name='fc7')
         .fc(101, name='fc8'))

    @property
    def data_path(self):
        # TODO ver si esto se puede poner en otra parte para no usar default_paths en el framework principal
        return paths.get_vgg16_weights_path()

    @property
    def has_transfer_learning(self):
        return True

    @property
    def trainable_layers(self):
        return ["fc7", "fc8"]

    def load(self, data_path, session, train_layers=None):
        print("LOADD CORRECTO ")


class AlexNet(Network):
    def setup(self):
        """
        Creates a Neural Net with a AlexNet architecture. The input data must have been previously set in
         the constructor of the object as 'data'.
         E.g.:
            net = AlexNet({'data': input_tensor})

         Architecture taken from:
            http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf
        :return: None
        """
        (self.feed('data')
         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
         .lrn(2, 2e-05, 0.75, name='norm1')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
         .lrn(2, 2e-05, 0.75, name='norm2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 384, 1, 1, name='conv3')
         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
         .fc(2048, name='fc6')
         .dropout(keep_prob=0.5,name="dp1")
         .fc(1024, name='fc7')
         .dropout(keep_prob=0.5, name="dp2")
         .fc(101, relu=False, name='fc8'))

    @property
    def data_path(self):
        # TODO ver si esto se puede poner en otra parte para no usar default_paths en el framework principal
        return paths.get_alexnet_weights_path()

    @property
    def has_transfer_learning(self):
        return True

    @property
    def trainable_layers(self):
        return ["fc6","fc7", "fc8"]


class CifarTFNet(Network):
    def setup(self):
        """
         Architecture taken from:
            https://www.tensorflow.org/tutorials/images/deep_cnn#model_training
        :return: None
        """
        (self.feed('data')
         .conv(5, 5, 64, 1, 1, padding='VALID', name='conv1')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
         .lrn(4, (0.001 / 9.0), 0.75, name='norm1')
         .conv(5, 5, 64, 1, 1, name='conv2')
         .lrn(4, (0.001 / 9.0), 0.75, name='norm2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .fc(384, name='fc3')
         .dropout(0.6, name="drop4")
         .fc(192, name='fc5')
         .fc(10, relu=False, name='fc6'))
