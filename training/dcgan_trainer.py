"""
Module for the training algorithm that uses artificial sampling with DCGAN
"""
import numpy as np
import tensorflow as tf

from libs.DCGAN_tensorflow.model import DCGAN
from training.dcgan_config import DCGANConfig
from training.trainer import Trainer
import libs.DCGAN_tensorflow.utils as dc_utils
from utils import constants as const

# TODO implementar
flags = tf.flags
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS


class DCGANTrainer(Trainer):
    """
    Trainer that uses the algorithm presented in "Evolutive deep models for online learning on data
    streams with no storage"
    See: http://ceur-ws.org/Vol-1958/IOTSTREAMING2.pdf
    """

    def _create_loss(self, tensor_y: tf.Tensor, net_output: tf.Tensor):
        return tf.losses.softmax_cross_entropy(tensor_y, net_output)

    def _create_optimizer(self, config: DCGANConfig, loss: tf.Tensor, var_list=None):
        if config.model_name == const.DATA_MNIST:
            self.dcgan = DCGAN(
                tf.get_default_session(),
                input_width=config.input_width,
                input_height=config.input_height,
                output_width=config.output_width,
                output_height=config.output_height,
                batch_size=config.train_configurations[0].batch_size,
                sample_num=config.train_configurations[0].batch_size,
                y_dim=10,
                c_dim=config.c_dim,
                z_dim=config.z_dim,
                dataset_name=config.model_name,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir)
        else:
            self.dcgan = DCGAN(
                tf.get_default_session(),
                input_width=config.input_width,
                input_height=config.input_height,
                output_width=config.output_width,
                output_height=config.output_height,
                batch_size=config.train_configurations[0].batch_size,
                sample_num=config.train_configurations[0].batch_size,
                z_dim=config.z_dim,
                dataset_name=config.model_name,
                crop=FLAGS.crop,
                checkpoint_dir=FLAGS.checkpoint_dir)

        self.dcgan.create_optimizers(config.dcgan_lr, config.beta1)
        dc_utils.show_all_variables()
        # TODO preferiblemente no guardar a config
        self.config = config
        return tf.train.RMSPropOptimizer(config.learn_rate).minimize(loss, var_list=var_list)

    def _train_batch(self, sess, image_batch, target_batch, tensor_x: tf.Tensor, tensor_y: tf.Tensor,
                     train_step: tf.Operation, loss: tf.Tensor, increment: int, iteration: int, total_it: int):

        # TODO arreglar esto: no debe preguntarse de forma explícita
        if len(target_batch) == self.config.train_configurations[0].batch_size:
            if FLAGS.train:
                self.dcgan.train(self.config.train_configurations[0].batch_size, self.config.model_name,
                                 image_batch, target_batch, increment, iteration)
            else:
                if not self.dcgan.load(FLAGS.checkpoint_dir)[0]:
                    raise Exception("[!] Train a model first, then run test mode")
        return 0, 0
