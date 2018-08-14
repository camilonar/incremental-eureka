"""
Module with Exceptions
"""


class OptionNotSupportedError(Exception):
    """
    This error is raised when the Option that is required isn't found. This can be an Optimizer requested by the user,
    a Dataset, or any kind of value that isn't supported in the current version of the program
    """

    def __init__(self, message):
        super().__init__(message)


class TestNotPreparedError(Exception):
    """
    This error is raised when the Optimizer that is required isn't found
    """

    def __init__(self, message):
        super().__init__(message)
