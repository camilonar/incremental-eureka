"""
Module containing various useful neural networks models
"""
import tensorflow as tf
from libs.caffe_tensorflow.network import Network


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
         .conv(1, 1, 101, 1, 1, name='mlp8')
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
         .fc(200, relu=False, name='fc8'))


class FastNet(Network):
    def setup(self):
        """
            to 256 d 256 images

        :return: None
        """
        (self.feed('data')
         .batch_normalization(name="BN1")
         .conv(3, 3, 64, 1, 1, padding='SAME', name='conv2')
         .batch_normalization(name="BN3")
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv4')
         .batch_normalization(name="BN5")
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv6')
         .batch_normalization(name="BN7")
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv8')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool9')
         .batch_normalization(name="BN10")
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv11')
         .batch_normalization(name="BN12")
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv13')
         .batch_normalization(name="BN14")
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv15')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool6')
         .batch_normalization(name="BN17")
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv18')
         .batch_normalization(name="BN19")
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv20')
         .batch_normalization(name="BN21")
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv22')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool23')
         .batch_normalization(name="BN24")
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv25')
         .batch_normalization(name="BN26")
         .conv(3, 3, 128, 1, 1, padding='SAME', name='conv27')
         .max_pool(2, 2, 2, 2, padding='SAME', name='pool28')
         .batch_normalization(name="BN29")
         .conv(1, 1, 128, 1, 1, padding='SAME', name='conv30')
         .batch_normalization(name="BN31")
         .conv(1, 1, 128, 1, 1, padding='SAME', name='conv32')
         .batch_normalization(name="BN33")
         .conv(1, 1, 100, 1, 1, padding='SAME', name='conv34')
         .avg_pool(2, 2, 2, 2, name="pol35"))

    def get_output(self):
        return tf.squeeze(self.terminals[-1])


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
         .fc(4096, name='fc6')
         .fc(4096, name='fc7')
         .fc(101, name='fc8'))


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
         .fc(4096, name='fc6')
         .fc(4096, name='fc7')
         .fc(10, relu=False, name='fc8'))


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
