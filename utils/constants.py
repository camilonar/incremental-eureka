"""
This module has constants used across the program
"""

# -------------Names for the different datasets------------

DATA_MNIST = "MNIST"
"""
Name for MNIST
"""
DATA_FASHION_MNIST = "FASHION-MNIST"
"""
Name for Fashion MNIST
"""
DATA_CIFAR_10 = "CIFAR-10"
"""
Name for CIFAR-10
"""
DATA_CIFAR_100 = "CIFAR-100"
"""
Name for CIFAR-100
"""
DATA_CALTECH_101 = "CALTECH-101"
"""
Name for CALTECH 101
"""
DATA_TINY_IMAGENET = "TINY IMAGENET"
"""
Name for Tiny Imagenet
"""
DATA_CALTECH_256 = "CALTECH-256"
"""
Name for CALTECH 256
"""

# -------------Names for the trainers (algorithms)------------

TR_BASE = "TR_BASE"
"""
Name for the trainer that uses RMSProp
"""
TR_DCGAN = "TR_DCGAN"
"""
Name for the trainer that uses DCGAN for artificial sampling
"""
TR_REP = "TR_REP"
"""
Name for the trainer that uses our proposed algorithm
"""

# -------------Names for the experiments------------

SUMMARY_INTERVAL = 500
"""
Default value for the interval (Number of iterations) at which validation/testing are performed
"""
CKP_INTERVAL = 2000
"""
Default value for the interval (Number of iterations) at which checkpoints are being saved
"""
SEED = 12345
"""
Default value for the seed for random values
"""
IS_INCREMENTAL = True
"""
It tells whether or not incremental learning is being used (useful for some Experiments)
"""
