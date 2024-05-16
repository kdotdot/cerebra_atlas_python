#!/usr/bin/env python
"""
Transforms
"""
from gzip import GzipFile
from typing import Tuple
import numpy as np
from mne.transforms import (
    Transform,
    combine_transforms,
    invert_transform,
    apply_trans as mne_apply_trans,
)


def _get_mgz_header(fname: str) -> dict:
    """Adapted from nibabel to quickly extract header info."""
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


def read_mri_info(path: str, units: str = "m") -> tuple:
    """Get tranforms from .mgz file"""
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
    """Move volume from Left Inferior Anterior to Right Anterior Superior space"""
    flipped_volume = np.rot90(volume, -1, axes=(1, 2))
    flipped_volume = np.flipud(flipped_volume)
    return flipped_volume


def lia_to_ras(volume: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Move lia volume from Left Inferior Anterior to Right Anterior Superior space
    Slso adjust and return affine

    Args:
        volume (np.ndarray): np.ndarray of shape (h, w, d)
        affine (np.ndarray): Affine matrix for volume

    Returns:
        Tuple[np.ndarray, np.ndarray]: Transformed volume and affine
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
    """Convert LIA points to RAS points"""
    ras_pts = lia_pts.copy()
    ras_pts[:, 0] = 255 - ras_pts[:, 0]  # LIA to RIA
    ras_pts[:, 1] = 255 - ras_pts[:, 1]  # RIA to RSA
    ras_pts[:, 1], ras_pts[:, 2] = ras_pts[:, 2], ras_pts[:, 1].copy()  # RSA to RAS
    return ras_pts


def apply_trans(trans: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Apply transformation to data"""
    return mne_apply_trans(trans, data)


def apply_inverse_trans(trans: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Apply inverse transformation to data"""
    return mne_apply_trans(np.linalg.inv(trans), data)
