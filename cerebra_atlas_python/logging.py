import logging
from typing import Union


def setup_logging(
    level: Union[str, int] = logging.DEBUG,
    mne_log_level: str = "WARNING",
    plt_log_level: str = "ERROR",
) -> None:
    """
    Sets up logging for the application with specified logging levels for different modules.

    Args:
        level (Union[str, int]): The logging level for the main logger. Can be a string (e.g., 'DEBUG', 'INFO')
                                  or an integer as defined in the logging module.
        mne_log_level (str): The logging level for the MNE module, specified as a string.
        plt_log_level (str): The logging level for the matplotlib module, specified as a string.

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

    logger = logging.getLogger()
    logger.setLevel(level=level)
    logging.basicConfig(
        level=level,
        format=" [%(levelname)s] %(asctime)s.%(msecs)02d %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    import mne

    # Setting the logging level for MNE
    mne.set_log_level(mne_log_level)  # type: ignore

    import matplotlib.pyplot as plt

    # Setting the logging level for matplotlib
    plt.set_loglevel(plt_log_level.lower())

    logging.getLogger("pyprep.reference").setLevel(logging.WARNING)
    logging.getLogger("PngImagePlugin").setLevel(logging.WARNING)
