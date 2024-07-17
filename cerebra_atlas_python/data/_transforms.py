#!/usr/bin/env python
"""
Transforms
"""
from gzip import GzipFile
from typing import Tuple, Dict
import numpy as np
from mne.transforms import (
    Transform,
    combine_transforms,
    invert_transform,
    apply_trans as mne_apply_trans,
)


def _get_mgz_header(fname: str) -> Dict[str, np.ndarray]:
    """Adapted from nibabel to quickly extract header info. from .mgz files

    Args:
        fname (str): Path to the MGZ file.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing header information.

    Raises:
        OSError: If the filename does not end with '.mgz'.
    """
    if not fname.endswith(".mgz"):
        raise OSError("Filename must end with .mgz")
    header_dtd = [
        ("version", ">i4"),
        ("dims", ">i4", (4,)),
        ("type", ">i4"),
        ("dof", ">i4"),
        ("goodRASFlag", ">i2"),
        ("delta", ">f4", (3,)),
        ("Mdc", ">f4", (3, 3)),
        ("Pxyz_c", ">f4", (3,)),
    ]
    header_dtype = np.dtype(header_dtd)
    with GzipFile(fname, "rb") as fid:
        hdr_str = fid.read(header_dtype.itemsize)
    header = np.ndarray(shape=(), dtype=header_dtype, buffer=hdr_str)
    # dims
    dims = header["dims"].astype(int)
    dims = dims[:3] if len(dims) == 4 else dims
    # vox2ras_tkr
    delta = header["delta"]
    ds = np.array(delta, float)
    ns = np.array(dims * ds) / 2.0
    v2rtkr = np.array(
        [
            [-ds[0], 0, 0, ns[0]],
            [0, 0, ds[2], -ns[2]],
            [0, -ds[1], 0, ns[1]],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    # ras2vox
    d = np.diag(delta)
    pcrs_c = dims / 2.0
    mdc = header["Mdc"].T
    pxyz_0 = header["Pxyz_c"] - np.dot(mdc, np.dot(d, pcrs_c))
    M = np.eye(4, 4)
    M[0:3, 0:3] = np.dot(mdc, d)
    M[0:3, 3] = pxyz_0.T
    header = dict(dims=dims, vox2ras_tkr=v2rtkr, vox2ras=M, zooms=header["delta"])
    return header


def read_mri_info(
    path: str, units: str = "m"
) -> Tuple[Transform, Transform, Transform]:
    """Get transforms from an MGZ file.

    Args:
        path (str): Path to the MGZ file.
        units (str, optional): Units for the transforms. Defaults to "m" (meters).

    Returns:
        Tuple[Transform, Transform, Transform]:
            - vox_ras_t: Transform from voxel to RAS (non-zero origin) space.
            - vox_mri_t: Transform from voxel to MRI space.
            - mri_ras_t: Transform from MRI to RAS (non-zero origin) space.

    Raises:
        ValueError: If `units` is not "m" or "mm".
    """
    hdr = _get_mgz_header(path)
    n_orig = hdr["vox2ras"]
    t_orig = hdr["vox2ras_tkr"]
    # dims = hdr["dims"]
    # zooms = hdr["zooms"]

    # extract the MRI_VOXEL to RAS (non-zero origin) transform
    vox_ras_t = Transform("mri_voxel", "ras", n_orig)

    # extract the MRI_VOXEL to MRI transform
    vox_mri_t = Transform("mri_voxel", "mri", t_orig)

    # construct the MRI to RAS (non-zero origin) transform
    mri_ras_t = combine_transforms(invert_transform(vox_mri_t), vox_ras_t, "mri", "ras")

    assert units in ("m", "mm")
    if units == "m":
        conv = np.array([[1e-3, 1e-3, 1e-3, 1]]).T
        # scaling and translation terms
        vox_ras_t["trans"] *= conv
        vox_mri_t["trans"] *= conv
        # just the translation term
        mri_ras_t["trans"][:, 3:4] *= conv

    return (vox_ras_t, vox_mri_t, mri_ras_t)  # , dims, zooms


def volume_lia_to_ras(volume: np.ndarray) -> np.ndarray:
    """Move volume from Left Inferior Anterior to Right Anterior Superior space

    Args:
    volume (np.ndarray): Input volume in LIA space.

    Returns:
        np.ndarray: Transformed volume in RAS space.
    """
    flipped_volume = np.rot90(volume, -1, axes=(1, 2))
    flipped_volume = np.flipud(flipped_volume)
    return flipped_volume


def lia_to_ras(volume: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Move a volume from Left Inferior Anterior (LIA) to Right Anterior Superior (RAS) space
    and adjust the affine matrix accordingly.

    Args:
        volume (np.ndarray): Input volume in LIA space.
        affine (np.ndarray): Affine matrix for the input volume.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Transformed volume in RAS space.
            - Adjusted affine matrix for the transformed volume.
    """
    flipped_volume = volume_lia_to_ras(volume)

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


def lia_points_to_ras_points(lia_pts: np.ndarray) -> np.ndarray:
    """Convert points from Left Inferior Anterior (LIA) to Right Anterior Superior (RAS) space.

    Args:
        lia_pts (np.ndarray): Input points in LIA space.

    Returns:
        np.ndarray: Transformed points in RAS space.
    """
    ras_pts = lia_pts.copy()
    ras_pts[:, 0] = 255 - ras_pts[:, 0]  # LIA to RIA
    ras_pts[:, 1] = 255 - ras_pts[:, 1]  # RIA to RSA
    ras_pts[:, 1], ras_pts[:, 2] = ras_pts[:, 2], ras_pts[:, 1].copy()  # RSA to RAS
    return ras_pts


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
    assert (merged_grid != 0).sum() == (grid1 != 0).sum() + (
        grid2 != 0
    ).sum(), (
        f"{(grid1 != 0).sum()= } {(grid2 != 0).sum()= } {(merged_grid != 0).sum()= }"
    )

    return merged_grid


def move_volume_from_ras_to_lia(volume: np.ndarray):
    """
    Transforms a volume from RAS  orientation to LIA orientation.

    """
    volume = np.flipud(volume)
    volume = np.rot90(volume, 1, axes=(1, 2))

    return volume


def apply_trans(trans: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Apply transformation to data

    Args:
    trans (np.ndarray): Transformation matrix.
    data (np.ndarray): Input data.

    Returns:
        np.ndarray: Transformed data.
    """
    return mne_apply_trans(trans, data)


def apply_inverse_trans(trans: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Apply inverse transformation to data

    Args:
    trans (np.ndarray): Transformation matrix.
    data (np.ndarray): Input data.

    Returns:
        np.ndarray: Transformed data.
    """
    return mne_apply_trans(np.linalg.inv(trans), data)
