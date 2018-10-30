"""
This module has constants used across the program
"""
import os

"""
Names for the datasets
"""
DATA_MNIST = "MNIST"
DATA_CIFAR_10 = "CIFAR-10"
DATA_CIFAR_100 = "CIFAR-100"
DATA_CALTECH_101 = "CALTECH-101"
DATA_TINY_IMAGENET = "TINY IMAGENET"
DATA_CALTECH_256 = "CALTECH-256"

"""
Names for the trainers (algorithms)
"""
TR_BASE = "TR_BASE"
TR_DCGAN = "TR_DCGAN"
TR_REP = "TR_REP"

"""
Default parameters for test
"""
SUMMARY_INTERVAL = 500
CKP_INTERVAL = 2000
SEED = 12345
IS_INCREMENTAL = False

"""
Default location for datasets
"""
__DATASET_PATH = os.path.join("..", "datasets")
MNIST_PATH = os.path.join(__DATASET_PATH, "MNIST")
CIFAR_10_PATH = os.path.join(__DATASET_PATH, "cifar10")
CIFAR_100_PATH = os.path.join(__DATASET_PATH, "cifar-100")
CALTECH_101_PATH = os.path.join(__DATASET_PATH, "101_ObjectCategories")
TINY_IMAGENET_PATH = os.path.join(__DATASET_PATH, "tiny-imagenet-200")
CALTECH_256_PATH = os.path.join(__DATASET_PATH, "256_ObjectCategories")
