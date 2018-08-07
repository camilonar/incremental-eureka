"""
Module with Exceptions
"""


class OptimizerNotSupportedError(Exception):
    """
    This error is raised when the Optimizer that is required isn't found
    """

    def __init__(self, message):
        super().__init__(message)


class TestNotPreparedError(Exception):
    """
    This error is raised when the Optimizer that is required isn't found
    """

    def __init__(self, message):
        super().__init__(message)
