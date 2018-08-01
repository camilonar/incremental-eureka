import numpy as np

class TrainingConfiguration:
    """A class for the configuration of the training of a 
    Neural Network
    """
    
    def __init__(self, tf, x, y, x_test, y_test):
        """It allows the creation of a new TrainingConfiguration.
        tf is the tensorflow module used in the main thread,
        x is the input placeholder tensor
        y is the output layer of the network 
        x_test and y_test are the test dataset inputs and outputs respectively
        """
        n_outputs = y.get_shape()[1]
        self.x = x
        self.y_ = tf.placeholder(tf.float32, [None, n_outputs]) #Represents the desired output
        self.x_weights = tf.placeholder(tf.float32, [None, n_outputs]) #Represents the weights of the instances
        self.y = y
        self.mse = tf.reduce_mean(tf.square(self.y - self.y_) * self.x_weights)
        self.train_step = tf.train.AdamOptimizer(0.004).minimize(self.mse)
        self.x_test = x_test
        self.y_test = y_test
    
    def __train_net(self, tf, sess, iterations, x_batch, y_batch, x_weights_values):
        """It trains the net with the previously configured parameters and also
        test the performance of the net against the test dataset
        """
        for i in range(iterations):
            _, c = sess.run([self.train_step, self.mse], feed_dict={self.x: x_batch, self.y_: y_batch, self.x_weights: x_weights_values})
            if i % 10 == 0:
                print(i * 100 / iterations, " val-> OK")
        
        test_results, _ = self.test_dataset(tf, sess, x_batch, y_batch, self.x_test, self.y_test)
        return test_results
    
    def run_basic_test(self, tf, sess, iterations, x_batch, y_batch):
        """It will run a test with the defined session, iterations and batch.
        The test runs a basic version of a training algorithm, i.e. there's no
        special method of training
        """
        x_weights_values = np.full((len(x_batch), self.y.get_shape()[1]), 1)
        
        print("\n:::PROBANDO:::::::::::\n")
        return self.__train_net(tf, sess, iterations, x_batch, y_batch, x_weights_values)
        
    def run_prototypes_test(self, tf, sess, iterations, x_batch, y_batch, x_frontiers, y_frontiers, x_kernels, y_kernels):
        """It will run a test with the defined session, iterations and batch.
        The test runs a a version of the algorithm with prototype selections
        The frontiers should be instances where the neural net is unsure
        of the classification, while the kernels should be the opposite
        """
        x_weights_values = np.full((len(x_batch), self.y.get_shape()[1]), 1)
        x_weights_values = np.append(x_weights_values, np.full((len(x_frontiers) + len(x_kernels), self.y.get_shape()[1]), 3), axis=0)
        x_batch_aux = np.concatenate((x_batch, np.asanyarray(x_frontiers)), axis=0)
        x_batch_aux = np.concatenate((x_batch_aux, np.asanyarray(x_kernels)), axis=0)
        y_batch_aux = np.concatenate((y_batch, np.asanyarray(y_frontiers)), axis=0)
        y_batch_aux = np.concatenate((y_batch_aux, np.asanyarray(y_kernels)), axis=0)
        
        print("\n:::PROBANDO PROTOTIPOS:::::::::::\n")
        return self.__train_net(tf, sess, iterations, x_batch_aux, y_batch_aux, x_weights_values)
    
    def test_dataset(self, tf, sess, x_input_v, y_input_v, x_test_v, y_test_v):
        """Test the neural network against the training dataset and a test 
        dataset to review performance
        """
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        test_result = 1.0 - sess.run(accuracy, feed_dict={self.x: x_test_v, self.y_: y_test_v})
        train_result = 1.0 - sess.run(accuracy, feed_dict={self.x: x_input_v, self.y_: y_input_v})

        print("Resultados Test: " + str(test_result))
        print("Resultados Training: " + str(train_result))

        return test_result, train_result