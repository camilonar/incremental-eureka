from libs.caffe_tensorflow.network import Network


class GoogleNet(Network):
    def setup(self):
        """
        Creates a Neural Net with GoogleNet architecture. The input must have been previously set in the constructor of
        the object as 'data'. E.g.:
            net = LeNet({'data': input_tensor})

        The implementation is based on the structure present in:
            https://storage.googleapis.com/pub-tools-public-publication-data/pdf/43022.pdf

        With the modifications from:
            http://cs231n.stanford.edu/reports/2017/pdfs/931.pdf
        :return: None
        """
        """(self.feed('data')
         .conv(7, 7, 64, 2, 2, padding='SAME', name='conv1')
         .max_pool(3, 3, 2, 2, padding='SAME', name='pool1')
         .lrn(2, 2e-05, 0.75, name='norm1')
         .conv(1, 1, 64, 1, 1, padding='VALID', name='conv2')
         .conv(3, 3, 192, 1, 1, padding='SAME', name='conv3')
         .lrn(2, 2e-05, 0.75, name='norm2')
         .max_pool(3, 3, 2, 2, padding='SAME', name='pool2')
         .inception_layer(64, 96, 128, 16, 32, 32, name="inception3a")
         .inception_layer(128, 128, 192, 32, 96, 64, name="inception3b")
         .max_pool(3, 3, 2, 2, padding='SAME', name='pool3')
         .inception_layer(192, 96, 208, 16, 48, 64, name="inception4a")
         .inception_layer(160, 112, 224, 24, 64, 64, name="inception4b")
         .inception_layer(128, 128, 256, 24, 64, 64, name="inception4c")
         .inception_layer(112, 144, 288, 32, 64, 64, name="inception3d")
         .inception_layer(256, 160, 320, 32, 128, 128, name="inception4e")
         .max_pool(3, 3, 2, 2, padding='SAME', name='pool4')
         .inception_layer(256, 160, 320, 32, 128, 128, name="inception5a")
         .inception_layer(384, 192, 384, 48, 128, 128, name="inception5b")
         .avg_pool(7, 7, 1, 1, padding='VALID', name='pool5')
         .dropout(keep_prob=0.4, name="dp1")
         .fc(101, relu=False, name='fc1'))"""
        (self.feed('data')
         .conv(7, 7, 32, 2, 2, padding='SAME', name='conv1')
         .max_pool(3, 3, 2, 2, padding='SAME', name='pool1')
         .lrn(2, 2e-05, 0.75, name='norm1')
         .conv(1, 1, 32, 1, 1, padding='VALID', name='conv2')
         .conv(3, 3, 96, 1, 1, padding='SAME', name='conv3')
         .lrn(2, 2e-05, 0.75, name='norm2')
         .max_pool(3, 3, 2, 2, padding='SAME', name='pool2')
         .inception_layer(32, 48, 64, 8, 16, 16, name="inception3a")
         .inception_layer(64, 64, 96, 16, 48, 32, name="inception3b")
         .max_pool(3, 3, 2, 2, padding='SAME', name='pool3')
         .inception_layer(96, 48, 104, 8, 24, 32, name="inception4a")
         .inception_layer(80, 56, 112, 12, 32, 32, name="inception4b")
         .inception_layer(64, 64, 128, 12, 32, 32, name="inception4c")
         .inception_layer(56, 72, 144, 16, 32, 32, name="inception3d")
         .inception_layer(128, 80, 160, 16, 64, 64, name="inception4e")
         .max_pool(3, 3, 2, 2, padding='SAME', name='pool4')
         .inception_layer(128, 80, 160, 16, 64, 64, name="inception5a")
         .inception_layer(192, 96, 192, 24, 64, 64, name="inception5b")
         .avg_pool(7, 7, 1, 1, padding='VALID', name='pool5')
         .dropout(keep_prob=0.4, name="dp1")
         .fc(200, relu=False, name='fc1'))
