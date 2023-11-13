"""
Various util functions
"""

import requests
import logging
import time
import numpy as np
import mne
import matplotlib.pyplot as plt
from configparser import ConfigParser, InterpolationMissingOptionError
import os
import os.path as op
from typing import Tuple, Dict, Optional, Any


def read_config_as_dict(
    file_path: str = op.dirname(__file__) + "/config.ini", section: Optional[str] = None
) -> Tuple[Dict[str, str], bool]:
    """
    Reads a configuration file and returns its contents as a dictionary.

    This function reads the specified configuration file and parses its contents.
    If a specific section is requested, only that section is returned. Otherwise,
    all sections are returned. Additionally, environment variables are used as
    default values.

    Args:
        file_path (str): Path to the configuration file. Defaults to 'config.ini' in the current directory.
        section (Optional[str]): Specific section to read from the configuration file.
                                    If None, all sections are read. Defaults to None.

    Returns:
        Tuple[Dict[str, Dict[str, str]], bool]: A tuple containing a dictionary of configuration values
                                                and a boolean indicating the success of reading the file.
                                                The dictionary is structured with sections as keys and
                                                dictionaries of the section's key-value pairs as values.
    """

    # ALLOW MISSING ENV VARIABLES (env variables which are empty throw InterpolationMissingOptionError)
    def attempt_get_value(parser: ConfigParser, section: str, key: str) -> str:
        try:
            return parser[section][key]
        except InterpolationMissingOptionError:
            return ""

    config_dict: Dict[str, Dict[str, Any]] = {}
    success: bool = True

    if not op.exists(file_path):
        logging.warning("Config file does not exist: %s", file_path)
        success = False
        return config_dict, success

    config_parser = ConfigParser()
    config_parser.read_dict({"DEFAULT": os.environ})
    config_parser.read(file_path)

    if section is None:
        for section in config_parser.sections():
            config_dict[section] = {}
            for key in config_parser[section]:
                if key.upper() not in os.environ:
                    config_dict[section][key] = attempt_get_value(
                        config_parser, section, key
                    )

        if not config_dict:
            logging.warning("Attempted to read empty config file: %s", file_path)
            success = False
    else:
        if section not in config_parser.sections():
            logging.warning(
                "Section '{section}' does not exist in config file: %s", file_path
            )
            success = False
        else:
            config_dict[section] = {}
            for key in config_parser[section]:
                if key.upper() not in os.environ:
                    config_dict[section][key] = attempt_get_value(
                        config_parser, section, key
                    )
            config_dict = config_dict[section]

    return config_dict, success


def time_func_decorator(func):
    def wrapper_function(*args, **kwargs):
        logging.info(f'{"*" * 10} START({func.__name__}) {"*" * 10}')
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        logging.info(
            f'{"*" * 10} END({func.__name__}) {"*" * 10} ({end_time-start_time:.2f} s)'
        )
        return res

    return wrapper_function


# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def setup_logging(level=logging.DEBUG, mne_log_level="WARNING", plt_log_level="ERROR"):
    levels = {
        "DEBUG": logging.DEBUG,
        "ERROR": logging.ERROR,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
    }
    if type(level) == str:
        assert level in levels.keys()
        level = levels[level]
    logger = logging.getLogger()
    logger.setLevel(level=level)
    logging.basicConfig(
        level=level,
        format=" [%(levelname)s] %(asctime)s.%(msecs)02d %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # MNE
    mne.set_log_level(mne_log_level)

    # PLT
    plt.set_loglevel(plt_log_level.lower())


def move_volume_from_LIA_to_RAS(volume, _affine=None):
    volume = np.rot90(volume, -1, axes=(1, 2))
    volume = np.flipud(volume)
    if _affine is None:
        return volume

    affine = _affine.copy()
    # Switch from LIA to RIA
    affine[0, -1] = 126  # Fix translation
    affine[0, 0] = 1

    # Switch from RIA to RSA
    affine[1, -1] = 256 - affine[2, -1]
    affine[2, 1] = 1

    # Switch from RSA to RAS
    aff = affine.copy()
    aff[1, :] = affine[2, :]
    aff[2, :] = affine[1, :]
    # affine[1, :], affine[2, :] = affine[2, :], affine[1, :] how?

    return volume, aff


# For a particular volume, check around every point and, if neighbor
# is empty (0), propagate value from original point
@time_func_decorator
def expand_volume(volume, perimeter=2):
    logging.info(f"{volume.shape= }")
    expand_pts = np.zeros((volume.shape[0], volume.shape[1], volume.shape[2]))
    # trace = np.zeros(256)
    for x in range(50, 200):
        for y in range(25, 225):
            zs = volume[x, y, :]
            if len(zs[zs != 0]) == 0:
                continue
            # print(expand_pts[x, y : y + perimeter, :].shape)
            # return

            indices = np.argwhere(zs != 0)
            z_start = int(indices[0])
            z_end = int(indices[-1])
            expand_pts[
                x - perimeter : x + perimeter,
                y - perimeter // 4 : y + perimeter // 4,
                z_start:z_end,
            ] = 103

            # expand_pts[x : x + perimeter, y, :][:, zs != 0] = np.repeat(
            #     np.expand_dims(zs, 1), perimeter, axis=1
            # ).T[:, zs != 0]

            # expand_pts[x, y - perimeter : y, :][:, zs != 0] = np.repeat(
            #     np.expand_dims(zs, 1), perimeter, axis=1
            # ).T[:, zs != 0]

            # expand_pts[x - perimeter : x, y, :][:, zs != 0] = np.repeat(
            #     np.expand_dims(zs, 1), perimeter, axis=1
            # ).T[:, zs != 0]

            # expand_pts[x, y : y - perimeter, :][:, zs != 0] = np.repeat(
            #     np.expand_dims(zs, 1), perimeter, axis=1
            # ).T[:, zs != 0]

            # expand_pts[x:+perimeter, y, :][:, zs != 0] = np.repeat(
            #     np.expand_dims(zs, 1), perimeter, axis=1
            # ).T[:, zs != 0]

            # expand_pts[x:-perimeter, y, :][:, zs != 0] = np.repeat(
            #     np.expand_dims(zs, 1), perimeter, axis=1
            # ).T[:, zs != 0]
            # expand_pts[x, y + 1, :] = zs
            # expand_pts[x, y - 1, :] = zs
            # expand_pts[x + 1, y, :] = zs
            # expand_pts[x - 1, y, :] = zs
            # expand_pts[x + 1, y + 1, :] = zs
            # expand_pts[x - 1, y - 1, :] = zs
            # expand_pts[x + 1, y + 1, :] = zs
            # expand_pts[x - 1, y - 1, :] = zs
            # print(volume[x, y, :].shape)
            # return
            # trace[zs != 0] = perimeter
            # volume[x, y : y + perimeter, :][:, zs != 0] = zs[zs != 0]
            # print(volume[x, y : y + perimeter, :][:, zs != 0].shape, zs[zs != 0])

    # expand_indices = (
    #     np.indices((perimeter, perimeter, perimeter)).reshape((3, -1)) - 1
    # ).T
    # pts = np.argwhere(volume != 0)

    # print(pts.shape)

    # for pt in pts:
    #     x, y, z = pt

    #     # val = volume[x, y, z]
    #     # if val == 0:
    #     #     continue

    #     # Look around point and expand

    #     for n in range(len(expand_indices)):
    #         i, j, k = expand_indices[n]
    #         # val_n = volume[x + i, y + j, z + k]  # value from neighbor
    #         # if val_n == 0:
    #         expand_pts[x + i, y + j, z + k] = 103

    # # Propagate
    volume[expand_pts != 0] = expand_pts[expand_pts != 0]
    return volume


def find_closest_point(points, target_point):
    # Convert the points array and target point to numpy arrays if they aren't already
    points = np.asarray(points)
    target_point = np.asarray(target_point)

    # Calculate the difference between each point in the array and the target point
    differences = points - target_point

    # Calculate the Euclidean distance for each point in the array
    distances = np.linalg.norm(differences, axis=1)

    # Find the index of the closest point
    closest_point_index = np.argmin(distances)

    # Return the closest point
    return points[closest_point_index], distances[closest_point_index]


def read_config_from_file(file_path: str):
    print(file_path)
    file_name = file.split("/")[-1][:-2]
    print(file_path, file_name)
    return read_config_as_dict(section=file_name)


def create_default_object(class_name):
    pass


def get_volume_ras():
    pass


if __name__ == "__main__":
    from pprint import pprint

    # file_id = "13rfrvxVQe18ss2hccPy10DkKQdnNyjWL"
    # destination = "./cer.mgz"
    # download_file_from_google_drive(file_id, destination)
    # path = "./config.ini"
    config = read_config_as_dict()
    # # my_config_parser_dict = {s: dict(config.items(s)) for s in config.sections()}
    pprint(config)

    config = read_config_as_dict(section="MNIAverage")
    # pprint(config)
    pprint(config)
