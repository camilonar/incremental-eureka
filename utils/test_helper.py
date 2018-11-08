"""
Module with useful functions for performing a test
"""
import utils.constants as const
from experiments_impl.experiments import Experiments


def print_config(dataset: str, optimizer: str, checkpoint_key: str, s_interval: int, ckp_interval: int, seed: int,
                 is_incremental: bool, train_dirs: [str], validation_dir: str, is_menu: bool = True):
    """
        Prints the configuration selected by the user
        :param dataset: a string representing the dataset that has been configured by the user
        :param optimizer: a string representing the optimizer that has been configured by the user
        :param checkpoint_key: a string representing a checkpoint. Must be None if no checkpoint has been configured
        :param s_interval: the summary interval that has been configured by the user
        :param ckp_interval: the checkpoint interval that has been configured by the user
        if the dataset doesn't have any dataset-specific path.
        :param seed: the seed for random numbers
        :param is_incremental: True to indicate that the training is gonna contain multiple mega-batches
        :param train_dirs: array of strings corresponding to the paths of each one of the mega-batches for training
        :param validation_dir: a string corresponding to the path of the testing data
        :param is_menu: boolean value. If set to True the menu version is used (asks for extra input)
    """
    print("\n--------------------------------------------------------")
    print("Starting test with the following configuration:\n")
    print("-Dataset: {}".format(dataset))
    print("-Optimizer: {}".format(optimizer))
    print("-Checkpoint: {}".format(checkpoint_key))
    print("-Summary interval: {} iterations".format(s_interval))
    print("-Checkpoint interval: {} iterations".format(ckp_interval))
    print("-Seed: {}".format(seed))
    print("-The test is {}".format("INCREMENTAL " if is_incremental else "NOT incremental"))
    print("-Training data directories: \n\t{}".format(train_dirs))
    print("-Validation data directory: \n\t{}".format(validation_dir))
    print("--------------------------------------------------------\n")

    if is_menu:
        input("To continue with the test press any key...")


def perform_test(dataset: str, optimizer: str, checkpoint_key: str, s_interval: int, ckp_interval: int, seed: int,
                 is_incremental: bool, train_dirs: [str], validation_dir: str, extras: [str]):
    """
    Prepares and performs the test according to the configuration given by the user
    :param dataset: a string representing the dataset that has been configured by the user
    :param optimizer: a string representing the optimizer that has been configured by the user
    :param checkpoint_key: a string representing a checkpoint. Must be None if no checkpoint has been configured
    :param s_interval: the summary interval that has been configured by the user
    :param ckp_interval: the checkpoint interval that has been configured by the user
    :param seed: the seed for random numbers
    :param is_incremental: True to indicate that the training is gonna contain multiple mega-batches
    :param train_dirs: array of strings corresponding to the paths of each one of the mega-batches for training
    :param validation_dir: a string corresponding to the path of the testing data
    :param extras: an array of strings corresponding to paths specific for each dataset. It should be an empty array
    if the dataset doesn't have any dataset-specific path.
    :return: None
    """
    const.SEED = seed
    factory = Experiments.get_experiment(optimizer, dataset)
    tester = factory(train_dirs, validation_dir, extras, s_interval, ckp_interval, checkpoint_key)

    tester.prepare_all(optimizer, is_incremental)
    tester.execute_test()
