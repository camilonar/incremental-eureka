from libs.caffe_tensorflow.network import Network


class ResNet18(Network):
    def setup(self, num_outputs):
        """
         ResNet18 based on the implementation in:
            https://github.com/fcipollone/TinyImageNet/blob/master/code/ti_classifiers.py

        :return: None
        """
        (self.feed('data')
         .conv(3, 3, 64, 1, 1, padding='SAME', name='conv1')
         .batch_normalization(relu=True, name='norm1')
         .residual_layer(64, name='res1')
         .residual_layer(64, name='res2')
         .residual_layer(128, stride=2, name='res4')
         .residual_layer(128, name='res5')
         .residual_layer(256, stride=2, name='res8')
         .residual_layer(256, name='res9')
         .residual_layer(512, stride=2, name='res14')
         .residual_layer(512, name='res15')
         .adaptive_avg_pool(1, 1, padding='VALID', name='pool2')
         .fc(num_outputs, relu=False, name='fc6'))
