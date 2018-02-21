import argparse
import os
import sys


import dataset as dt
import utils as ut
import numpy as np
import nnet as nn
import tensorflow as tf

FLAGS = None	
	
def train():

	def probar_dataset(sess, x_input_v, y_input_v, x_test_v, y_test_v):
		#Test
		matriz_y = y_test_v
		pred = sess.run(y,feed_dict={x:x_test_v , y_:matriz_y})
		errors = 0 
		for i in range (pred.shape[0]):
			if(np.argmax(pred[i]) != np.argmax(matriz_y[i])):errors=errors+1

		print ("Porcentaje error test dataset  : "+str((errors/pred.shape[0])*100))
		
		#Train
		aux2 = y_input_v
		pred = sess.run(y,feed_dict={x:x_input_v , y_:aux2})
		errors = 0 
		for i in range (pred.shape[0]):
			if(np.argmax(pred[i]) != np.argmax(aux2[i])):errors=errors+1

		print ("Porcentaje error train dataset  : "+str((errors/pred.shape[0])*100))

	s=np.asarray([1,0,0])
	ve=np.asarray([0,1,0])
	vi=np.asarray([0,0,1])
	dict = {}
	dict = ut.add_element_to_dict(dict, 'species', {'setosa': s, 'versicolor': ve,'virginica': vi})
	train, test = dt.prepare_data_from_csv("iris.csv",100, dict)
	x_input, y_input = dt.divide_x_and_y(train,4)
	x_test, y_test = dt.divide_x_and_y(test,4)

	#El squeeze se hace porque originalmente quedaba un array de 3 dimensiones, con una de las dimensiones teniendo sólo 1 objeto.  
	y_input = np.squeeze(y_input)
	y_test = np.squeeze(y_test)

	#El stack se hace para que no que como array de arrays, sino como matriz (o un array multidimensional)
	y_input = np.stack(y_input)
	y_test = np.stack(y_test)

	n_hidden = 250 

	x= tf.placeholder(tf.float32,[None,4]) #representa la entrada
	y_=tf.placeholder(tf.float32,[None,3]) #representa la salida deseada

	hidden1 = nn.nn_layer(x, 4, n_hidden, 'layer1', tf.nn.tanh)
	y = nn.nn_layer(hidden1, n_hidden, 3, 'layer2', tf.nn.tanh)

	mse  = tf.reduce_mean(tf.square(y - y_))
	train_step = tf.train.AdamOptimizer(0.005).minimize(mse) 
	
	# Merge all the summaries and write them out to
	# /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
	merged = tf.summary.merge_all()
	sess = tf.InteractiveSession()
	train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
	test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

	init = tf.global_variables_initializer()
	sess.run(init)


	for i in range(500):#gradiente descendente estocastico
		if i % 10 == 0:  # Record summaries and test-set accuracy
		  summary, acc = sess.run([merged, mse], feed_dict={x:x_input , y_:y_input})
		  test_writer.add_summary(summary, i)
		  print('Accuracy at step %s: %s' % (i, acc))
		else:  # Record train set summaries, and train
			if i % 100 == 99:  # Record execution stats
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
				summary, _ = sess.run([merged, train_step],
									  feed_dict={x:x_input , y_:y_input},
									  options=run_options,
									  run_metadata=run_metadata)
				train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
				train_writer.add_summary(summary, i)
				print('Adding run metadata for', i)
			else:  # Record a summary
				summary, _ = sess.run([merged, train_step], feed_dict={x:x_input , y_:y_input})
				train_writer.add_summary(summary, i)
			
	probar_dataset(sess, x_input, y_input, x_test, y_test)
	
	train_writer.close()
	test_writer.close()

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory for storing input data')
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/mnist_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

