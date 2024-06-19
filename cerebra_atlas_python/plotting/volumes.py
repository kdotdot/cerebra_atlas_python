import logging
from typing import Optional, Tuple
import numpy as np


logger = logging.getLogger(__name__)


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
    logger.debug(
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


def get_ax_labels(axis: int) -> tuple[int, int]:
    """
    Determines the x and y axis labels based on the provided axis.

    This function takes an integer representing an axis (0, 1, or 2) and returns
    a tuple of integers representing the x and y labels. The labels are determined
    as follows:
    - If axis is 0, x_label is 1 and y_label is 2.
    - If axis is 1, x_label is 0 and y_label is 2.
    - If axis is 2, x_label is 0 and y_label is 1.

    Parameters:
    axis (int): An integer representing the axis (expected to be 0, 1, or 2).

    Returns:
    tuple[int, int]: A tuple containing two integers representing the x and y labels.
    """
    if axis == 0:
        x_label = 1
        y_label = 2
    elif axis == 1:
        x_label = 0
        y_label = 2
    elif axis == 2:
        x_label = 0
        y_label = 1
    else:
        raise ValueError("axis must be 0, 1, or 2")
    return x_label, y_label


def project_volume_2d(
    volume_slice,
    axis=0,
    colors=None,
    alpha_values=None,
    size_values=None,
    avoid_values=None,
):
    avoid_values = avoid_values or [0]
    x_label, y_label = get_ax_labels(axis)

    mask = ~np.isin(volume_slice, avoid_values)
    xyzs = np.where(mask)
    xs_ys = np.array([xyzs[x_label], xyzs[y_label]])

    # FILTER_DUPLICATES
    xs_ys, unique_indices = np.unique(xs_ys, axis=1, return_index=True)

    xyzs = np.take(xyzs, unique_indices, axis=1)

    new_values = np.array(volume_slice[tuple(xyzs)]).astype(int)
    cs = colors[new_values] if colors is not None else None
    alphas = alpha_values[new_values] if alpha_values is not None else None
    sizes = size_values[new_values] if size_values is not None else None

    return xs_ys.T, cs, alphas, sizes


def merge_points_optimized(
    xs_ys_arr: Tuple[Optional[np.ndarray], np.ndarray],
    cs_arr: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    alphas_arr: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    sizes_arr: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    default_color: Optional[list] = None,
    default_alpha: float = 1,
    default_size: float = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Merges two sets of points, colors, and alpha values while removing duplicates.

    """
    default_color = default_color or [
        1,
        0,
        1,
    ]
    xs_ys_keep, xs_ys_new = xs_ys_arr
    cs_keep, cs_new = cs_arr
    alphas_keep, alphas_new = alphas_arr
    sizes_keep, sizes_new = sizes_arr

    if alphas_new is None:
        alphas_new = np.full(len(xs_ys_new), default_alpha)
    if cs_new is None:
        cs_new = np.tile(default_color, (len(xs_ys_new), 1))
    if sizes_new is None:
        sizes_new = np.full(len(xs_ys_new), default_size)
    if xs_ys_keep is None:
        return xs_ys_new, cs_new, alphas_new, sizes_new

    # Step 1: Use a hash-based approach to identify non-duplicate points
    keep_set = set(map(tuple, xs_ys_keep))
    non_dup_indices = [
        i for i, point in enumerate(xs_ys_new) if tuple(point) not in keep_set
    ]
    non_dup_xs_ys_new = xs_ys_new[non_dup_indices]

    # Step 2: Efficiently handle color and alpha arrays
    if cs_keep is None:
        cs_keep = np.tile(default_color, (len(xs_ys_keep), 1))
    if cs_new is not None:
        cs_new = cs_new[non_dup_indices]  # Index the cs_new list

    if alphas_keep is None:
        alphas_keep = np.full(len(xs_ys_new), default_alpha)
    if alphas_new is not None:
        alphas_new = alphas_new[non_dup_indices]  # Index the alphas_new list

    if sizes_keep is None:
        sizes_keep = np.full(len(xs_ys_new), default_size)
    if sizes_new is not None:
        sizes_new = sizes_new[non_dup_indices]  # Index the alphas_new list

    # Step 3: Merge arrays
    xs_ys = np.vstack((xs_ys_keep, non_dup_xs_ys_new))
    cs = np.vstack((cs_keep, cs_new)) if cs_new is not None else cs_keep
    alphas = (
        np.concatenate((alphas_keep, alphas_new))
        if alphas_new is not None
        else alphas_keep
    )
    sizes = (
        np.concatenate((sizes_keep, sizes_new)) if sizes_new is not None else sizes_keep
    )

    return xs_ys, cs, alphas, sizes
