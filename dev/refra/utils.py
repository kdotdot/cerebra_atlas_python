"""
Various common util functions
"""

import logging
import time
from typing import Optional, List, Tuple, Callable, Any, Union

import nibabel as nib
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

# from cerebra_atlas_python.plotting import get_cmap_colors_hex


def time_func_decorator(func: Callable) -> Callable:
    """
    A decorator that logs the start and end time of the execution of a function.
    It also logs the total duration of the function's execution.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The wrapper function which includes the logging functionality.
    """

    def wrapper_function(*args: Any, **kwargs: Any) -> Any:
        logging.info("%s START(%s) %s", "*" * 10, func.__name__, "*" * 10)
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        logging.info(
            "%s END(%s) %s (%.2f s)",
            "*" * 10,
            func.__name__,
            "*" * 10,
            end_time - start_time,
        )
        return res

    return wrapper_function

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

    # Setting the logging level for MNE
    mne.set_log_level(mne_log_level)

    # Setting the logging level for matplotlib
    plt.set_loglevel(plt_log_level.lower())

    logging.getLogger('pyprep.reference').setLevel(logging.WARNING)
    logging.getLogger('PngImagePlugin').setLevel(logging.WARNING)

def lia_to_ras(volume, affine=None):
    flipped_volume = np.rot90(volume, -1, axes=(1, 2))
    flipped_volume = np.flipud(flipped_volume)
    
    if affine is None:
        return flipped_volume

    flip_matrix = np.eye(4)
    flip_matrix[0, 0] = -1  # Flip X
    flip_matrix[1, 2] = -1  # Flip Z
    flip_matrix[1, 1] = 0  # Transpose Z
    flip_matrix[2, 2] = 0  # Transpose Y
    flip_matrix[2, 1] = 1  # Transpose Z

    # Adjust the translation part of the affine for the flip in X and Z axes
    flip_matrix[0, 3] = volume.shape[0] - 1
    flip_matrix[1, 3] = volume.shape[2] - 1

    # Compute the new affine matrix
    new_affine = np.dot(affine, flip_matrix)
    
    return flipped_volume, new_affine

# def move_volume_from_lia_to_ras(
#     volume: np.ndarray, affine: Optional[np.ndarray] = None
# ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
#     """
#     Transforms a volume from LIA (Left Inferior Anterior) orientation to RAS (Right Anterior Superior) orientation.

#     Args:
#         volume (np.ndarray): The input volume in LIA orientation.
#         affine (Optional[np.ndarray]): An optional affine transformation matrix associated with the volume.
#                                         If provided, it will be modified to reflect the change in orientation.

#     Returns:
#         Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: The transformed volume in RAS orientation.
#                                                          If an affine matrix is provided, returns a tuple of the
#                                                          transformed volume and the modified affine matrix.
#     """
#     volume = np.rot90(volume, -1, axes=(1, 2))
#     volume = np.flipud(volume)
#     if affine is None:
#         return volume

#     affine = affine.copy()
#     # Switch from LIA to RIA
#     affine[0, -1] = 126  # Fix translation
#     affine[0, 0] = 1

#     # Switch from RIA to RSA
#     affine[1, -1] = 256 - affine[2, -1]
#     affine[2, 1] = 1

#     # Switch from RSA to RAS
#     affine[1:3, :] = np.roll(affine[1:3, :], 1, axis=0)
#     # affine[1, :], affine[2, :] = affine[2, :], affine[1, :] how?

#     return volume, affine


def move_volume_from_ras_to_lia(volume: np.ndarray):
    """
    Transforms a volume from RAS  orientation to LIA orientation.

    """
    volume = np.flipud(volume)
    volume = np.rot90(volume, 1, axes=(1, 2))

    return volume


def find_closest_point(
    points: List[List[float]], target_point: List[float]
) -> Tuple[np.ndarray, float]:
    """
    Finds the closest point to a given target point from a list of points.

    Args:
        points (List[List[float]]): A list of points, each point is a list of coordinates.
        target_point (List[float]): The target point as a list of coordinates.

    Returns:
        Tuple[np.ndarray, float]: A tuple containing the closest point and its Euclidean distance from the target point.
    """
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


def slice_volume(
    volume: np.ndarray, fixed_value: int, axis: int = 0, n_layers: int = 1
) -> np.ndarray:
    """
    Slices a given volume array along a specified axis.

    Args:
        volume (np.ndarray): The input volume array.
        fixed_value (int): The starting value for slicing.
        axis (int): The axis along which to slice the volume. Defaults to 0.
        n_layers (int): The number of layers to include in the slice. Defaults to 1.

    Returns:
        np.ndarray: The sliced volume array.
    """
    start_slice, end_slice = fixed_value, fixed_value + n_layers
    increment = 1
    logging.debug(
        "start_slice=%s  end_slice=%s  increment=%s ", start_slice, end_slice, increment
    )
    slice_idx = slice(start_slice, end_slice, increment)
    if axis == 0:
        return volume[slice_idx, :, :]
    elif axis == 1:
        return volume[:, slice_idx, :]
    elif axis == 2:
        return volume[:, :, slice_idx]
    else:
        raise ValueError(f"Invalid axis: {axis}")


def rgb_to_hex_str(color_rgb: np.ndarray) -> str:
    """Transforms (r,g,b) (0,1) array into hex color string

    Args:
        color_rgb (np.ndarray): input array

    Returns:
        str: transformed hex string
    """
    color_rgb = [int(c * 255) for c in color_rgb]
    return f"#{color_rgb[0]:02x}{color_rgb[1]:02x}{color_rgb[2]:02x}"


def merge_voxel_grids(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """
    Merges two voxel grids of the same size into one.

    The merge operation is a logical OR between the corresponding elements of the two voxel grids.
    If a voxel is set in either of the grids (value of 1), it will be set in the resulting merged grid.

    Args:
        grid1 (np.ndarray): The first voxel grid, expected to be of size [256, 256, 256].
        grid2 (np.ndarray): The second voxel grid, expected to be of the same size as grid1.

    Returns:
        np.ndarray: The merged voxel grid of size [256, 256, 256].

    Raises:
        ValueError: If the input grids are not of the same shape.
    """
    if grid1.shape != grid2.shape:
        raise ValueError("The two voxel grids must be of the same shape.")

    # Perform logical OR operation to merge the voxel grids
    merged_grid = grid1 + grid2

    # Make sure total number of voxels stayed the same
    # (No voxels overlap)
    assert (merged_grid != 0).sum() == (grid1 != 0).sum() + (grid2 != 0).sum(),f"{(grid1 != 0).sum()= } {(grid2 != 0).sum()= } {(merged_grid != 0).sum()= }"

    return merged_grid


def point_cloud_to_voxel(
    point_cloud: np.ndarray, dtype=np.uint8, vox_value: int = 1
) -> np.ndarray:
    """
    Transforms a given point cloud into a voxel array.

    This function takes an array representing a point cloud where each point is a 3D coordinate,
    and converts it into a voxel representation with a specified size of [256, 256, 256].
    Each point in the point cloud is mapped to a voxel in this 3D grid. The values in the point
    cloud array should be in the range [0, 257) (RAS).

    Args:
        point_cloud (np.ndarray): A numpy array of shape [n_points, 3] representing the point cloud,
                                  where n_points is the number of points in the cloud and each point
                                  is a 3D coordinate.
        dtype (type, optional): The data type to be used for the voxel grid. Defaults to np.uint8.
        vox_value (int): set value for voxel grid

    Returns:
        np.ndarray: A voxel array of shape [256, 256, 256] representing the 3D grid, where each
                    element is set to 1 if it corresponds to a point in the input array, otherwise 0.
    """
    # Initialize a voxel grid of the specified size filled with zeros
    voxel_grid = np.zeros((256, 256, 256), dtype=dtype)

    # Iterate through each point in the point cloud
    for point in point_cloud:
        # Check if the point is within the valid range
        if all(0 <= coord < 256 for coord in point):
            # Convert the floating point coordinates to integers
            x, y, z = map(int, point)
            # Set the corresponding voxel to 1
            voxel_grid[x, y, z] = vox_value

    return voxel_grid


# def point_cloud_to_voxel_lia(
#     point_cloud: np.ndarray, dtype=np.uint8, vox_value: int = 1
# ) -> np.ndarray:
#     # Initialize a voxel grid of the specified size filled with zeros
#     voxel_grid = np.zeros((256, 256, 256), dtype=dtype)

#     # Iterate through each point in the point cloud
#     for point in point_cloud:
#         # Check if the point is within the valid range
#         if all(0 <= coord < 256 for coord in point):
#             # Convert the floating point coordinates to integers
#             l, i, a = map(int, point)
#             # Set the corresponding voxel to 1
#             voxel_grid[x, y, z] = vox_value

#     return voxel_grid


# Helper functions
def get_standard_montage(kept_ch_names=None, kind="GSN-HydroCel-129", head_size=0.1025):
    original_montage = mne.channels.make_standard_montage(kind, head_size=head_size)
    new_montage = original_montage.copy()

    ind = [
        i
        for (i, channel) in enumerate(original_montage.ch_names)
        if channel
        in (kept_ch_names if kept_ch_names is not None else original_montage.ch_names)
    ]
    # Keep only the desired channels
    new_montage.ch_names = [original_montage.ch_names[x] for x in ind]
    kept_channel_info = [original_montage.dig[3:][x] for x in ind]
    # Keep the first three rows as they are the fiducial points information
    new_montage.dig = new_montage.dig[:3] + kept_channel_info  #

    return new_montage


def inspect_img(path):
    img = nib.load(path)
    data = img.get_fdata()
    img.orthoview()
    logging.info(f"{img.shape= }")
    codes = nib.orientations.aff2axcodes(img.affine)
    logging.info(f"Coordinate frame: {''.join(codes)}")
    logging.info(f"\n{img.affine}")
    return img, data


def get_volume_ras(path, dtype=np.uint8):
    """
    Loads a medical image volume from the given path and converts its coordinate frame from LIA to RAS.

    This function:
    1. Loads the volume using nibabel.
    2. Converts the volume's coordinate frame from LIA (Left, Inferior, Anterior) to RAS (Right, Anterior, Superior)
       using the move_volume_from_lia_to_ras function.

    Args:
        path (str): The file path of the medical image volume.
        dtype (type, optional): The data type to be used for the volume data. Defaults to np.uint8.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the transformed volume data and its affine matrix.
    """
    img = nib.load(path)  # All volumes are in LIA coordinate frame
    volume, affine = lia_to_ras(
        np.array(img.dataobj, dtype=dtype), img.affine
    )
    return volume, affine


if __name__ == "__main__":
    pass
