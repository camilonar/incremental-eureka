import tensorflow as tf
import numpy as np

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """    
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations
	  
def create_neural_net(input, neurons_array, act=tf.nn.relu):
    """It allows the creation of a complete neural network.
    It receives an input tensor for the whole net, an array with the number 
    of neurons of each layer where the first position is the number of inputs
    (e.g. [30,10,20,10] will result in a neural net with 3 layers with the 
    structure 10-20-10 and with 30 inputs) and also the activation function 
    of each layer.
    If the activation function is not an array then the whole net will have
    the same activation function.
    It returns the output tensor of the neural net.
    """
    curr_act = act
    layers = []
    for i in range(len(neurons_array)-1):
        # If the user provided an array of activations is trated as such
        if (type(act) is list):
            curr_act = act[i]
            
        if i==0:
            input_tensor = input
        else:
            input_tensor = layers[i-1]
        layers = np.append(layers, nn_layer(input_tensor, neurons_array[i], neurons_array[i+1], 'layer'+str(i+1), curr_act)) 

    return layers[-1]