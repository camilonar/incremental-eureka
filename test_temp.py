import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import utils as utils
import models
from tests.ImagenetTester import ImagenetTester
from train_conf import TrainConfig, GeneralConfig
import trainer

tf.logging.set_verbosity(tf.logging.INFO)


def get_MNIST():
    mnist = input_data.read_data_sets("../datasets/MNIST_data/", one_hot=True, reshape=False)
    return mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


def test_dataset(x, y, y_, sess, x_test_v, y_test_v, accuracy):
    """Test the neural network against the training dataset and a test
    dataset to review performance
    """

    test_result = 1.0 - sess.run(accuracy, feed_dict={x: x_test_v, y_: y_test_v})

    print("% Error Test: " + str(test_result))

    return test_result


def test_tiny_imagenet():
    # Configuration
    config = GeneralConfig(0.9, 0.1, 0.1)
    ck_path, summaries_path = utils.prepare_directories(config)

    x_caffe = tf.placeholder(tf.float32, [None, 256, 256, 3])
    y_caffe = tf.placeholder(tf.float32, [None, 200])
    outputs = models.CaffeNet({'data': x_caffe})

    # Running the training
    iterations = 10
    sess = tf.InteractiveSession()
    graph = tf.Graph()
    with tf.device('/cpu:0'):
        from input.imagenet_data import ImagenetData

        d = ImagenetData(batch_size=128, sess=sess)
        image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()

    mse = tf.reduce_mean(tf.square(y_caffe - outputs.get_output()))
    correct_prediction = tf.equal(tf.argmax(y_caffe, 1), tf.argmax(outputs.get_output(), 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('mse', mse)
    train_step = tf.train.RMSPropOptimizer(0.001).minimize(mse)
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    with graph.as_default():
        train_writer = tf.summary.FileWriter(summaries_path + '/train',
                                             graph)
        for i in range(iterations):
            image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
            _, c = sess.run([train_step, mse],
                            feed_dict={x_caffe: image_batch, y_caffe: target_batch})
            if i % 2 == 0:
                print(i * 100 / iterations, c)
                summary, _ = sess.run([merged, mse],
                                      feed_dict={x_caffe: image_batch, y_caffe: target_batch})
                train_writer.add_summary(summary, i)
                test_dataset(x_caffe, outputs.get_output(), y_caffe, sess, image_batch, target_batch, accuracy)
    print("Finished")


if __name__ == '__main__':
    test_tiny_imagenet()

"""# Loading the data
train_x, train_y, test_x, test_y = get_MNIST()
train_x = train_x[:10000]
train_y = train_y[:10000]
train_x = np.pad(train_x, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
n_outputs = train_y.shape[1]
n_inputs = train_x.shape[1]
# Creating the net
x = tf.placeholder(tf.float32, [None, 32, 32, 1])
y = tf.placeholder(tf.float32, [None, n_outputs])
# inputs = tf.reshape(train_x, [-1, 28, 28, 1])
# outputs = models.LeNet({'data': x})"""
