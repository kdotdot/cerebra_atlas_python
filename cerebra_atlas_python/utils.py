"""
Various common util functions
"""

import logging
import time
from typing import Optional, List, Tuple, Callable, Any, Union

import numpy as np
import mne
import matplotlib.pyplot as plt


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


def move_volume_from_lia_to_ras(
    volume: np.ndarray, affine: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Transforms a volume from LIA (Left Inferior Anterior) orientation to RAS (Right Anterior Superior) orientation.

    Args:
        volume (np.ndarray): The input volume in LIA orientation.
        affine (Optional[np.ndarray]): An optional affine transformation matrix associated with the volume.
                                        If provided, it will be modified to reflect the change in orientation.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: The transformed volume in RAS orientation.
                                                         If an affine matrix is provided, returns a tuple of the
                                                         transformed volume and the modified affine matrix.
    """
    volume = np.rot90(volume, -1, axes=(1, 2))
    volume = np.flipud(volume)
    if affine is None:
        return volume

    affine = affine.copy()
    # Switch from LIA to RIA
    affine[0, -1] = 126  # Fix translation
    affine[0, 0] = 1

    # Switch from RIA to RSA
    affine[1, -1] = 256 - affine[2, -1]
    affine[2, 1] = 1

    # Switch from RSA to RAS
    affine[1:3, :] = np.roll(affine[1:3, :], 1, axis=0)
    # affine[1, :], affine[2, :] = affine[2, :], affine[1, :] how?

    return volume, affine


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


# Modified _make_volume_source_space from https://github.com/mne-tools/mne-python/blob/maint/1.6/mne/source_space/_source_space.py#L1640-L1945
def get_neighbors(source_space: mne.SourceSpaces) -> List[np.ndarray]:
    return []
    # pass in cerebra.mni_average.bem["surfs"][2]["rr"]
    # must be converted to meters (is already)

    mins = np.min(rr, axis=0)
    maxs = np.max(rr, axis=0)

    print(f"{source_space= } {mins= } {maxs= }")

    grid = 5  # ?

    maxn = np.array(
        [
            np.floor(np.abs(m) / grid) + 1 if m > 0 else -np.floor(np.abs(m) / grid) - 1
            for m in maxs
        ],
        int,
    )
    minn = np.array(
        [
            np.floor(np.abs(m) / grid) + 1 if m > 0 else -np.floor(np.abs(m) / grid) - 1
            for m in mins
        ],
        int,
    )
    npts = source_space["inuse"]
    neigh = np.empty((26, npts), int)
    neigh.fill(-1)
    # Figure out each neighborhood:
    # 6-neighborhood first
    rr = source_space["rr"]
    x, y, z = rr[2].ravel(), rr[1].ravel(), rr[0].ravel()
    idxs = [
        z > minn[2],
        x < maxn[0],
        y < maxn[1],
        x > minn[0],
        y > minn[1],
        z < maxn[2],
    ]

    # Now make the initial grid
    ns = tuple(maxn - minn + 1)
    npts = np.prod(ns)
    nrow = ns[0]
    ncol = ns[1]
    nplane = nrow * ncol
    k = np.arange(npts)
    offsets = [-nplane, 1, nrow, -1, -nrow, nplane]
    for n, idx, offset in zip(neigh[:6], idxs, offsets):
        n[idx] = k[idx] + offset

    # Then the rest to complete the 26-neighborhood

    # First the plane below
    idx1 = z > minn[2]

    idx2 = np.logical_and(idx1, x < maxn[0])
    neigh[6, idx2] = k[idx2] + 1 - nplane
    idx3 = np.logical_and(idx2, y < maxn[1])
    neigh[7, idx3] = k[idx3] + 1 + nrow - nplane

    idx2 = np.logical_and(idx1, y < maxn[1])
    neigh[8, idx2] = k[idx2] + nrow - nplane

    idx2 = np.logical_and(idx1, x > minn[0])
    idx3 = np.logical_and(idx2, y < maxn[1])
    neigh[9, idx3] = k[idx3] - 1 + nrow - nplane
    neigh[10, idx2] = k[idx2] - 1 - nplane
    idx3 = np.logical_and(idx2, y > minn[1])
    neigh[11, idx3] = k[idx3] - 1 - nrow - nplane

    idx2 = np.logical_and(idx1, y > minn[1])
    neigh[12, idx2] = k[idx2] - nrow - nplane
    idx3 = np.logical_and(idx2, x < maxn[0])
    neigh[13, idx3] = k[idx3] + 1 - nrow - nplane

    # Then the same plane
    idx1 = np.logical_and(x < maxn[0], y < maxn[1])
    neigh[14, idx1] = k[idx1] + 1 + nrow

    idx1 = x > minn[0]
    idx2 = np.logical_and(idx1, y < maxn[1])
    neigh[15, idx2] = k[idx2] - 1 + nrow
    idx2 = np.logical_and(idx1, y > minn[1])
    neigh[16, idx2] = k[idx2] - 1 - nrow

    idx1 = np.logical_and(y > minn[1], x < maxn[0])
    neigh[17, idx1] = k[idx1] + 1 - nrow - nplane

    # Finally one plane above
    idx1 = z < maxn[2]

    idx2 = np.logical_and(idx1, x < maxn[0])
    neigh[18, idx2] = k[idx2] + 1 + nplane
    idx3 = np.logical_and(idx2, y < maxn[1])
    neigh[19, idx3] = k[idx3] + 1 + nrow + nplane

    idx2 = np.logical_and(idx1, y < maxn[1])
    neigh[20, idx2] = k[idx2] + nrow + nplane

    idx2 = np.logical_and(idx1, x > minn[0])
    idx3 = np.logical_and(idx2, y < maxn[1])
    neigh[21, idx3] = k[idx3] - 1 + nrow + nplane
    neigh[22, idx2] = k[idx2] - 1 + nplane
    idx3 = np.logical_and(idx2, y > minn[1])
    neigh[23, idx3] = k[idx3] - 1 - nrow + nplane

    idx2 = np.logical_and(idx1, y > minn[1])
    neigh[24, idx2] = k[idx2] - nrow + nplane
    idx3 = np.logical_and(idx2, x < maxn[0])
    neigh[25, idx3] = k[idx3] + 1 - nrow + nplane

    # Omit unused vertices from the neighborhoods
    logging.info("Adjusting the neighborhood info.")
    r0 = minn * grid
    voxel_size = grid * np.ones(3)
    ras = np.eye(3)
    neigh_orig = neigh
    for sp in sps:
        # remove non source-space points
        neigh = neigh_orig.copy()
        neigh[:, np.logical_not(sp["inuse"])] = -1
        # remove these points from neigh
        old_shape = neigh.shape
        neigh = neigh.ravel()
        checks = np.where(neigh >= 0)[0]
        removes = np.logical_not(np.isin(checks, sp["vertno"]))
        neigh[checks[removes]] = -1
        neigh.shape = old_shape
        neigh = neigh.T
        # Thought we would need this, but C code keeps -1 vertices, so we will:
        # neigh = [n[n >= 0] for n in enumerate(neigh[vertno])]
        sp["neighbor_vert"] = neigh

    return neigh


if __name__ == "__main__":
    pass
