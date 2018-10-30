"""
This module acts as a shell interface for performing the tests
"""
import argparse
import utils.constants as const
import utils.default_paths as paths
import utils.test_helper as helper


def create_parser():
    """
    Creates the argument Parser
    :return: an ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        help='The name of the dataset (as stored in constants.py).')
    parser.add_argument(
        '-o',
        '--optimizer',
        type=str,
        help='The name of the optimizer (as stored in constants.py).')
    parser.add_argument(
        '-ck',
        '--checkpoint_key',
        type=str,
        default=None,
        help='(Optional) The ID of a checkpoint (e.g. .../MNIST/model-0-2000.ckpt key is \"0-2000\").')
    parser.add_argument(
        '-si',
        '--summaries_interval',
        dest='s_interval',
        type=int,
        default=const.SUMMARY_INTERVAL,
        help='(Optional) The interval of iterations at which validation is going to be performed. If not set then a '
             'default value is used instead.')
    parser.add_argument(
        '-ci',
        '--checkpoints_interval',
        dest='ckp_interval',
        type=int,
        default=const.CKP_INTERVAL,
        help='(Optional) The interval of iterations at which checkpoints are going to be saved. If not set then a '
             'default value is used instead.')
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=const.SEED,
        help='(Optional) The seed for random numbers. If not set then a default value is used instead.')
    parser.add_argument(
        '-i',
        '--is_incremental',
        action="store_true",
        default=const.IS_INCREMENTAL,
        help='(Optional) Must be set for incremental training.')
    parser.add_argument(
        '-dp',
        '--dataset_path',
        type=str,
        default=None,
        help='(Optional) The path to the dataset root directory. If not set then the default paths are used instead'
             'for the supported datasets.')

    return parser


def unpack_variables(dataset: str, optimizer: str, checkpoint_key: str, s_interval: int, ckp_interval: int, seed: int,
                     is_incremental: bool, dataset_path: str):
    """
    Helper function that is used as a proxy for easily unpacking variables from the corresponding parser
    :param dataset: a string representing the dataset that has been configured by the user
    :param optimizer: a string representing the optimizer that has been configured by the user
    :param checkpoint_key: a string representing a checkpoint. Must be None if no checkpoint has been configured
    :param s_interval: the summary interval that has been configured by the user
    :param ckp_interval: the checkpoint interval that has been configured by the user
    :param seed: the seed for random numbers
    :param is_incremental: True to indicate that the training is gonna contain multiple mega-batches
    :param dataset_path: the path to the root of the dataset
    :return: the values of the unpacked arguments in the same order
    """
    return dataset, optimizer, checkpoint_key, s_interval, ckp_interval, seed, is_incremental, dataset_path


def main():
    """
        Executes the program
        :return: None
        """
    parser = create_parser()
    args = parser.parse_args()
    dataset, optimizer, checkpoint_key, s_interval, ckp_interval, seed, is_incremental, dataset_path = unpack_variables(
        **vars(args))
    train_dirs, validation_dir, extras = paths.get_paths_from_dataset(dataset, is_incremental, dataset_path)
    helper.print_config(dataset, optimizer, checkpoint_key, s_interval, ckp_interval, seed, is_incremental, train_dirs,
                        validation_dir)
    helper.perform_test(dataset, optimizer, checkpoint_key, s_interval, ckp_interval, seed, is_incremental,
                        train_dirs, validation_dir, extras)
    return 0


if __name__ == '__main__':
    main()
