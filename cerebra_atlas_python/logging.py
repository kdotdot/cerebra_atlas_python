"""Logging utils"""
import logging
import logging.config
from typing import Union

# Loggers for modules are initialized on imports
# This is a workaround to set the logging level for the MNE and matplotlib modules
# pylint: disable=unused-import
import mne
import matplotlib.pyplot as plt

def setup_logging(
    level: Union[str, int] = logging.DEBUG,
    imported_level: str = "ERROR",
) -> None:
    """
    Sets up logging for the application with specified logging levels for different modules.

    Args:
        level (Union[str, int]): The logging level for the main logger. Can be a string (e.g., 'DEBUG', 'INFO')
                                  or an integer as defined in the logging module.
        imported_level (str): The logging level for imported modules (plt,mne...)

    Raises:
        AssertionError: If the 'level' provided as a string is not one of the recognized logging levels.
    """
    levels = {
        "DEBUG": logging.DEBUG,
        "ERROR": logging.ERROR,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
    }
    if isinstance(level, str):
        assert level in levels, f"Unrecognized logging level: {level}"
        level = levels[level]


    # Setting the logging level for all existing loggers to error
    logging.getLogger().setLevel(imported_level)

    logger = logging.getLogger()
    logger.setLevel(level=level)
    logging.basicConfig(
        level=level,
        format=" [%(levelname)s] %(asctime)s.%(msecs)02d %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
