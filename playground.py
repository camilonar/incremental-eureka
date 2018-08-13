import tensorflow as tf
from tensorflow.data import Dataset
from train_conf import GeneralConfig
from train_conf import TrainConfig


sess = tf.InteractiveSession()
with tf.device('/cpu:0'):

   
    from input.cifar_data import CifarData
    generalConfig = GeneralConfig(0.4)
    trainconf = TrainConfig(epochs=10,batch_size=100)
    generalConfig.add_train_conf(train_conf = trainconf )

    d = CifarData(general_config = generalConfig )

    image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()
    test_data_tensor, test_labels_tensor = d.build_test_data_tensor()

#sess.run(d.iterator.initializer)
for i in range(5):
    image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
    print("Batch 1...")
    len(image_batch)
    print(target_batch)

for i in range(5):
    image_batch, target_batch = sess.run([test_data_tensor, test_labels_tensor])
    print("Batch 1...")
    len(image_batch)
    print(target_batch)

d.change_dataset_part(1)
image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()

for i in range(100):
    d.change_dataset_part(1)

for i in range(5):
    image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
    print("Batch 1...")
    len(image_batch)
    print(target_batch)
import time
    #while True:
    #    d.change_dataset_part(1)
    #
    #    image_batch_tensor, target_batch_tensor = d.build_train_data_tensor()
    #    image_batch, target_batch = sess.run([image_batch_tensor, target_batch_tensor])
    #    time.sleep(30)
d.close()
print("Finished")
exit(1)
