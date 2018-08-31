"""
This module acts as an interface for performing the tests
"""
import utils.constants as const
import utils.default_paths as paths
from tests.imagenet_tester import ImagenetTester
from tests.caltech_tester import CaltechTester
from tests.cifar_tester import CifarTester
from tests.mnist_tester import MnistTester

def print_menu():
    """
    Prints the menu for the configuration of a Test
    :return: a str, corresponding to the response of the user
    """
    print()
    print("--------------MENU FOR CONFIGURATION OF TESTS---------------------")
    print("[1] Select Dataset and associated Neural Network")
    print("[2] Select Optimizer")
    print("[3] Select Checkpoint to Load (Optional)")
    print("[4] Select Learning Rate (If it isn't provided then the default value will be used)")
    print("[5] Select Summary and Checkpoint interval (If not provided then the default values will be used)")
    print("[0] Finish configuration and execute Test")
    print("[X] Exit")
    print("------------------------------------------------------------------")
    print()
    response = input("Select an option: ")
    return response


def ask_for_configuration():
    """
    Asks the user for the following configurations for a test:
    -Dataset
    -Optimizer (between the supported types)
    -Load of a checkpoint (Optional)
    -Learning Rate (Optional)
    -Summary Interval (Optional)
    -Checkpoint Interval (Optional)
    :return: a tuple containing 3 strings, a float and 3 ints: name of the dataset (str), name of the optimizer (str),
    representation of a checkpoint (str), learning rate (float), summary interval (int) and checkpoint interval (int),
    in that order. The representation of a checkpoint will be returned as None if the user doesn't configure it.
    An example of a return value is: ('TINY_IMAGENET', 'BASE', None, 0.05, 100, 300).
    """

    # Creation of variables
    response = "s"
    dataset, optimizer, checkpoint = None, None, None
    learning_rate, summary_interval, ckp_interval = const.LEARNING_RATE, const.SUMMARY_INTERVAL, const.CKP_INTERVAL

    while response:
        response = print_menu()

        if response == "1":
            dataset = configure_dataset_and_neural_net(dataset)
        elif response == "2":
            optimizer = configure_optimizer(optimizer)
        elif response == "3":
            checkpoint = configure_checkpoint(checkpoint)
        elif response == "4":
            learning_rate = configure_learning_rate(learning_rate)
        elif response == "5":
            summary_interval, ckp_interval = configure_intervals(summary_interval, ckp_interval)
        elif response == "0":
            if dataset and optimizer:
                return dataset, optimizer, checkpoint, learning_rate, summary_interval, ckp_interval
            else:
                print("Both the Dataset and Optimizer must be configured before performing a test")
                continue
        elif response.upper() == 'X':
            break
        else:
            print("Option '{}' is not recognized.".format(response))
            continue
    print("Exiting...")


def configure_dataset_and_neural_net(curr_dataset: str):
    """
        Configures the optimizer appropriately
        :param curr_dataset: the current dataset that has been configured by the user
        :return: a str representing the new dataset if the user has changed it to a valid value, otherwise, the current
        dataset will be returned
        """
    print()
    print("----------------------------Configure the Dataset----------------------------")
    print("\n**Note** The dataset is associated with a fixed neural network. For example, if you choose Tiny \n"
          "Imagenet Dataset then CaffeNet will be automatically selected as Model for the training.\n")

    while True:
        print("[M] MNIST (uses LeNet)")
        print("[C] CIFAR-10 (uses AlexNet)")
        print("[L] CALTECH-101 (uses NiN)")
        print("[I] TINY IMAGENET (uses CaffeNet)")
        print("[X] Cancel Operation and return to Main Menu")
        response = input("Select an optimizer: ").upper()
        if response == 'M':
            response = const.DATA_MNIST
        elif response == 'C':
            response = const.DATA_CIFAR_10
        elif response == 'L':
            response = const.DATA_CALTECH_101
        elif response == 'I':
            response = const.DATA_TINY_IMAGENET
        elif response.upper() == 'X':
            break
        else:
            print("Option '{}' is not recognized.".format(response))
            continue
        print("The dataset has been set to {}".format(response))
        return response

    print("The dataset hasn't been changed and the current configured value is {}".format(curr_dataset))
    return curr_dataset


def configure_optimizer(curr_optimizer: str):
    """
    Configures the optimizer appropriately
    :param curr_optimizer: the current optimizer that has been configured by the user
    :return: a str representing the new optimizer if the user has changed it to a valid value, otherwise, the current
    optimizer will be returned
    """
    print()
    print("----------------------------Configure the Optimizer----------------------------")

    while True:
        print("[B] Simple RMSProp (Base optimizer)")
        print("[C] CEAL Algorithm (Active Learning)")
        print("[R] Incremental Representative Sampling (The proposed method)")  # TODO cambiar a un nombre más apropiado
        print("[X] Cancel Operation and return to Main Menu")
        response = input("Select an optimizer: ").upper()
        if response == 'B':
            response = const.TR_BASE
        elif response == 'C':
            response = const.TR_CEAL
        elif response == 'R':
            response = const.TR_REPRESENTATIVES
        elif response == 'X':
            break
        else:
            print("Option '{}' is not recognized.".format(response))
            continue
        print("The optimizer has been set to {}".format(response))
        return response

    print("The optimizer hasn't been changed and the current configured value is {}".format(curr_optimizer))
    return curr_optimizer


def configure_checkpoint(curr_ckp: str):
    """
    Configures a representation of a checkpoint based in the corresponding increment (mega-batch) and iteration given
    by the user. Notice that this function DOES NOT check if the given checkpoint is valid. Also, the full checkpoint
    path must be created at a later moment, since that path is heavily dependent of other parameters like Dataset and
    Optimizer that may change during configuration time.
    :param curr_ckp: a string representing the increment and iteration of a checkpoint. It must be in the form:
    "[increment]-[iteration]", e.g. "1-20"
    :return: a string representing the new checkpoint if the user has changed it, otherwise, the current checkpoint will
     be returned. A value of None will be returned if there was a checkpoint already configured and the user decided to
     not load a checkpoint anymore.
    """
    print()
    print("----------------------------Configure the Checkpoint----------------------------")
    while True:
        print("[1] Select a checkpoint to be loaded")
        print("[2] Reset checkpoint")
        print("[X] Cancel Operation and return to Main Menu")
        response = input("Select an option: ").upper()
        if response == '1':
            response = get_checkpoint()
        elif response == '2':
            response = reset_checkpoint(curr_ckp)
            return response
        elif response == 'X':
            break
        else:
            print("Option '{}' is not recognized.".format(response))
            continue
        print("The checkpoint has been changed and corresponds to increment {} and iteration {}."
              .format(response.split("-")[0], response.split("-")[1]))
        return response

    if curr_ckp:
        print("The checkpoint hasn't been changed and currently corresponds to increment {} and iteration {}."
              .format(curr_ckp.split("-")[0], curr_ckp.split("-")[1]))
    else:
        print("No checkpoint has been selected at this moment.")

    return curr_ckp


def get_checkpoint():
    """
    Ask the user for the values of increment and iteration for the checkpoint
    :return: a string representing the increment and iteration, with the format "[increment]-[iteration]", e.g. "0-50"
    """
    print()
    while True:
        response = input("Please enter the mega-batch and iteration corresponding to the desired checkpoint with the "
                         "format [mega-batch]-[iteration], e.g. '0-50' (both must be ints equal or greater than 0): ")
        if len(response.split("-")) == 2:
            if 0 <= int(response.split("-")[0]):
                if 0 <= int(response.split("-")[1]):
                    return response
                else:
                    print("Invalid value for the iteration. Must be a int greater or equal than 0")
            else:
                print("Invalid value for the mega-batch. Must be a int greater or equal than 0")
        else:
            print("The format of the checkpoint is [mega-batch]-[iteration], e.g. '0-50', '1-0'")


def reset_checkpoint(curr_ckp: str):
    """
    Resets the checkpoint
    :param curr_ckp: a string representing the increment and iteration of a checkpoint.
    :return: None if the checkpoint is reset, otherwise, the current checkpoint will be returned
    """
    print()

    while True:
        response = input(
            "Are you sure you want to reset the checkpoint? This will mean that no checkpoint will be loaded"
            "when the training starts [Y/N]: ").upper()
        if response == 'Y':
            print("The checkpoint has been reset")
            return None
        elif response == 'N':
            print("The checkpoint has NOT been reset")
            return curr_ckp
        else:
            print("Option '{}' is not recognized.".format(response))
            continue


def configure_learning_rate(curr_lr: float):
    """
    Configures the learning rate appropriately
    :param curr_lr: the current learning rate that has been configured by the user
    :return: a float representing the new learning rate if the user has changed it to a valid value, otherwise, the
    current learning rate will be returned
    """
    response = "s"
    print()
    print("----------------------------Configure the Learning Rate----------------------------")
    print("Press [X] if you want to cancel the operation")

    while True:
        response = input("Set a value for the learning rate (must be a floating value between 0 and 1): ")
        if response.upper() == "X":
            break
        try:
            lr = float(response)
            if 0 < lr < 1:
                print("The learning rate has been set to {}".format(lr))
                return lr
            else:
                print("Invalid value for the learning rate. Must be a float between 0 and 1, non-inclusive")
        except ValueError:
            print("Invalid value for the learning rate. Must be a float")

    print("The learning rate hasn't been changed and the current configured value is {}".format(curr_lr))
    return curr_lr


def configure_intervals(curr_s_interval: int, curr_ckp_interval: int):
    """
    Configures the summary interval and checkpoint interval
    :param curr_s_interval: the current summary interval that has been configured by the user
    :param curr_ckp_interval: the current checkpoint interval that has been configured by the user
    :return: a tuple with 2 integers: the new summary interval and the new checkpoint interval if the user has changed
    them to a valid value, otherwise, the current summary and checkpoint intervals will be returned
    """
    print()
    print("----------------------------Configure the Intervals----------------------------")
    print("Press [X] if you want to cancel the operation")

    while True:
        response = input("Set a value for the summary interval (must be a integer greater than 0): ")
        if response.upper() == "X":
            break
        try:
            s_interval = int(response)
            if 0 < s_interval:
                ckp_multiplier = get_checkpoint_multiplier()
                if ckp_multiplier > 0:
                    print("The summary interval has been set to {}".format(s_interval))
                    print("The checkpoint interval has been set to {}".format(s_interval * ckp_multiplier))
                    return s_interval, ckp_multiplier * s_interval
            else:
                print("Invalid value for the summary interval. Must be a int greater than 0")
        except ValueError:
            print("Invalid value for the summary interval. Must be an int")

    print("The summary interval hasn't been changed and the current configured value is {}".format(curr_s_interval))
    print(
        "The checkpoint interval hasn't been changed and the current configured value is {}".format(curr_ckp_interval))
    return curr_s_interval, curr_ckp_interval


def get_checkpoint_multiplier():
    """
    Ask the user for the multiplier for the checkpoint interval, since this interval must be a multiple of the summary
    interval
    :return: an int greater than 0 that corresponds to the multiplier selected by the User, or -1 if the User cancels
    the operation
    """
    print("\nPress [X] if you want to cancel the operation")
    print("\n The checkpoint interval is a multiplier of the summary interval. For example, if you set a multiplier \n"
          "of 3 and the summary interval is 200, then the checkpoint interval will be set to 600.")
    while True:
        response = input("Set a value for the checkpoint interval multiplier (must be a integer greater than 0): ")
        if response.upper() == "X":
            return -1
        try:
            ckp_multiplier = int(response)
            if 0 < ckp_multiplier:
                return ckp_multiplier
            else:
                print("Invalid value for the checkpoint interval multiplier. Must be a int greater than 0")
        except ValueError:
            print("Invalid value for the checkpoint interval multiplier. Must be an int")


def print_config(dataset: str, optimizer: str, checkpoint: str, lr: float, s_interval: int, ckp_interval: int):
    """
        Prints the configuration selected by the user
        :param dataset: a string representing the dataset that has been configured by the user
        :param optimizer: a string representing the optimizer that has been configured by the user
        :param checkpoint: a string representing a checkpoint. Must be None if no checkpoint has been configured
        :param lr: the learning rate that has been configured by the user
        :param s_interval: the summary interval that has been configured by the user
        :param ckp_interval: the checkpoint interval that has been configured by the user
        if the dataset doesn't have any dataset-specific path.
    """
    print("--------------------------------------------------------")
    print("\n\nStarting test with the following configuration:\n")
    print("Dataset: {}".format(dataset))
    print("Optimizer: {}".format(optimizer))
    print("Checkpoint: {}".format(checkpoint))
    print("Learning rate: {}".format(lr))
    print("Summary interval: {} iterations".format(s_interval))
    print("Checkpoint interval: {} iterations".format(ckp_interval))
    print("\n")

    input("To continue with the test press any key...")


def perform_test(dataset: str, optimizer: str, checkpoint: str, lr: float, s_interval: int, ckp_interval: int,
                 train_dirs: [str], validation_dir: str, extras: [str]):
    """
    Prepares and performs the test according to the configuration given by the user
    :param dataset: a string representing the dataset that has been configured by the user
    :param optimizer: a string representing the optimizer that has been configured by the user
    :param checkpoint: a string representing a checkpoint. Must be None if no checkpoint has been configured
    :param lr: the learning rate that has been configured by the user
    :param s_interval: the summary interval that has been configured by the user
    :param ckp_interval: the checkpoint interval that has been configured by the user
    :param train_dirs: array of strings corresponding to the paths of each one of the mega-batches for training
    :param validation_dir: a string corresponding to the path of the testing data
    :param extras: an array of strings corresponding to paths specific for each dataset. It should be an empty array
    if the dataset doesn't have any dataset-specific path.
    :return: None
    """
    tester = None

    if dataset == const.DATA_MNIST:
        tester = MnistTester(lr, train_dirs, validation_dir, extras, s_interval, ckp_interval, checkpoint)
    if dataset == const.DATA_CIFAR_10:
        tester = CifarTester(lr, train_dirs, validation_dir, extras, s_interval, ckp_interval, checkpoint)
    if dataset == const.DATA_CALTECH_101:
        tester = CaltechTester(lr, train_dirs, validation_dir, extras, s_interval, ckp_interval, checkpoint)
    if dataset == const.DATA_TINY_IMAGENET:
        tester = ImagenetTester(lr, train_dirs, validation_dir, extras, s_interval, ckp_interval, checkpoint)

    tester.prepare_all(optimizer)
    tester.execute_test()


# TODO permitir que el usuario escoja los paths para el dataset
def main():
    """
    Executes the program
    :return: None
    """
    dataset, optimizer, checkpoint, lr, s_interval, ckp_interval = ask_for_configuration()
    train_dirs, validation_dir, extras = paths.get_paths_from_dataset(dataset)
    print_config(dataset, optimizer, checkpoint, lr, s_interval, ckp_interval)
    perform_test(dataset, optimizer, checkpoint, lr, s_interval, ckp_interval, train_dirs, validation_dir, extras)


if __name__ == '__main__':
    main()
