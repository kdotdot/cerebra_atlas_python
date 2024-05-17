import numpy as np
from gzip import GzipFile
from mne.transforms import Transform, combine_transforms, invert_transform


def _get_mgz_header(fname):
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
    Mdc = header["Mdc"].T
    pxyz_0 = header["Pxyz_c"] - np.dot(Mdc, np.dot(d, pcrs_c))
    M = np.eye(4, 4)
    M[0:3, 0:3] = np.dot(Mdc, d)
    M[0:3, 3] = pxyz_0.T
    header = dict(dims=dims, vox2ras_tkr=v2rtkr, vox2ras=M, zooms=header["delta"])
    return header


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


def lia_points_to_ras_points(lia_pts):
    ras_pts = lia_pts.copy()
    ras_pts[:, 0] = 255 - ras_pts[:, 0]  # LIA to RIA
    ras_pts[:, 1] = 255 - ras_pts[:, 1]  # RIA to RSA
    ras_pts[:, 1], ras_pts[:, 2] = ras_pts[:, 2], ras_pts[:, 1].copy()  # RSA to RAS
    return ras_pts
