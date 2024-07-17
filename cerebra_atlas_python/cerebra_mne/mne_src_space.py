#!/usr/bin/env python
"""
SourceSpace
"""
import os.path as op
from functools import cached_property
import numpy as np
from cerebra_atlas_python.data._cache import cache_np, cache_mne_src
from cerebra_atlas_python.data._transforms import (
    point_cloud_to_voxel,
    volume_lia_to_ras,
    move_volume_from_ras_to_lia,
)
from cerebra_atlas_python.data.cerebra_data import CerebraData
from ..data._transforms import apply_trans, apply_inverse_trans


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
class SourceSpaceData:
    """Uses CerebraData to generate the source space mask and points"""

    def __init__(
        self,
        cache_path: str,
        cerebra_data: CerebraData,
        source_space_grid_size: int = 3,
        source_space_include_wm: bool = False,
        source_space_include_non_cortical: bool = False,
        **kwargs,
    ):
        """_summary_

        Args:
            cache_path (_type_, optional): Path to cerebra cache dir. Defaults to None.
            cerebra_data (cerebra_data): CerebraData object
            source_space_grid_size (int, optional): Grid size for generating the source space,
                bigger means more downsampling (less src space points). Defaults to 3.
            source_space_include_wm (bool, optional): Whether to include whitematter
                in the source space. Defaults to False.
            source_space_include_non_cortical (bool, optional):Whether to include non-cortical
                regions in the source space. Defaults to False.
            kwargs: Additional arguments to pass to CerebraData
        """

        self.cache_path = cache_path
        self.cerebra_data = cerebra_data

        self.source_space_grid_size: int = source_space_grid_size
        self.source_space_include_wm: bool = source_space_include_wm
        self.source_space_include_non_cortical: bool = source_space_include_non_cortical

        self.wm_str = "wm" if self.source_space_include_wm else ""
        self.nc_str = "_nc" if self.source_space_include_non_cortical else ""
        self.src_space_string = (
            f"src_space_{self.source_space_grid_size}mm{self.wm_str}{self.nc_str}"
        )
        self._src_space_mask_path = op.join(
            self.cache_path, f"{self.src_space_string}_mask.npy"
        )
        self._src_space_mask_lia_path = op.join(
            self.cache_path, f"{self.src_space_string}_lia_mask.npy"
        )
        self._src_space_points_path = op.join(
            self.cache_path, f"{self.src_space_string}_src_pts.npy"
        )
        self._src_space_points_path_lia = op.join(
            self.cache_path, f"{self.src_space_string}_src_pts_lia.npy"
        )

    @cached_property
    def src_space_mask(self) -> np.ndarray:
        """Mask for cerebra_volume in RAS space.
        Indicates whether point from (256,256,256) belongs to the source space

        Returns:
            np.ndarray: (256,256,256) mask in RAS space
        """
        return cache_np(self.get_source_space_mask, self._src_space_mask_path, "ras")

    @cached_property
    def src_space_mask_lia(self) -> np.ndarray:
        """Mask for cerebra_volume in LIA space.
        Indicates whether point from (256,256,256) belongs to the source space

        Returns:
            np.ndarray: (256,256,256) mask in LIA space
        """
        return cache_np(
            self.get_source_space_mask, self._src_space_mask_lia_path, "lia"
        )

    @cached_property
    def src_space_points_lia(self) -> np.ndarray:
        """Array of length N x 3 containing N points in LIA space

        Returns:
            np.ndarray: points in LIA space
        """

        def compute_fn(self):
            return np.indices([256, 256, 256])[:, self.src_space_mask_lia].T

        return cache_np(compute_fn, self._src_space_points_path_lia, self)

    @cached_property
    def src_space_points(self) -> np.ndarray:
        """Array of length N x 3 containing N points in RAS space

        Returns:
            np.ndarray: points in RAS space
        """

        def compute_fn(self):
            return np.indices([256, 256, 256])[:, self.src_space_mask].T

        return cache_np(compute_fn, self._src_space_points_path, self)

    @cached_property
    def src_space_labels(self) -> np.ndarray:
        """Array of length N containing label for each point in src_space_points

        Returns:
            np.ndarray: 1D array of region ids [0,103]. Length N.
        """
        return self.cerebra_data.cerebra_volume[self.src_space_mask]

    @cached_property
    def src_space_labels_lia(self) -> np.ndarray:
        """Array of length N containing label for each point in src_space_points

        Returns:
            np.ndarray: 1D array of region ids [0,103]. Length N.
        """
        return self.cerebra_data.cerebra_volume_lia[self.src_space_mask_lia]

    @cached_property
    def src_space_n_total_points(self) -> int:
        """
        Returns:
            int: The total number of points in the source space (N).
        """
        return len(self.src_space_points)

    # pylint: disable=too-many-locals
    def get_source_space_mask(self, coord_frame="lia") -> np.ndarray:
        """Get volume mask for source space in RAS or LIA space
        Uses:
        self.source_space_grid_size
        self.source_space_include_wm
        self.source_space_include_non_cortical

        Args:
            coord_frame (str, optional): "ras" or "lia" coordinate frame. Defaults to "ras".

        Raises:
            ValueError: Unkown coordinta frame. Use "ras" or "lia"

        Returns:
            np.ndarray: mask
        """

        volume, _ = self.cerebra_data._get_wm_filled_cerebra_volume_aff_lia()

        if coord_frame != "lia" and coord_frame != "ras":
            raise ValueError(f"Unknown {coord_frame=}")

        # Create grid (downsample available volume)
        size = self.source_space_grid_size
        grid = np.zeros((256, 256, 256))
        a = np.arange(size - 1, 256, size)
        for x in a:
            for y in a:
                grid[x, y, a] = 1
        grid_mask = grid.astype(bool)

        # If include non-cortical keep all
        if self.source_space_include_non_cortical:
            not_zero_mask = volume != self.cerebra_data.get_region_id_from_region_name(
                "Empty"
            )
            not_wm_mask = volume != self.cerebra_data.get_region_id_from_region_name(
                "White matter"
            )
            downsampled_not_zero_mask = np.logical_and(grid_mask, not_zero_mask)
            downsampled_not_wm_mask = np.logical_and(grid_mask, not_wm_mask)
            # Combined mask is not zero and not wm
            combined_mask = np.logical_and(
                downsampled_not_zero_mask, downsampled_not_wm_mask
            )

        else:  # Keep only cortical:
            # Get cortical region ids
            cortical_ids = self.cerebra_data.get_cortical_region_ids()
            combined_mask = np.zeros((256, 256, 256)).astype(bool)
            for c_id in cortical_ids:
                # Handle each region individually
                region_mask = volume == c_id
                # Downsample based on grid
                downsampled_region_mask = np.logical_and(grid_mask, region_mask)
                # Add region to mask
                combined_mask = np.logical_or(combined_mask, downsampled_region_mask)

        # Add whitematter if needed
        if self.source_space_include_wm:
            whitematter_mask = (
                volume
                == self.cerebra_data.get_region_id_from_region_name("White matter")
            )
            # Downsample
            downsampled_whitematter_mask = np.logical_and(grid_mask, whitematter_mask)
            combined_mask = np.logical_or(combined_mask, downsampled_whitematter_mask)

        if coord_frame == "ras":
            combined_mask = volume_lia_to_ras(combined_mask)

        return combined_mask

    def get_src_space_n_points_per_region(self, region_ids: np.ndarray) -> np.ndarray:
        """Get array of length len(region_ids) containing number of points in each region

        Args:
            region_ids (np.ndarray): Region ids to count points for []

        Returns:
            np.ndarray: 1D array with number of points in each region
        """
        points_per_region = np.zeros(len(region_ids))
        region_ids, counts = np.unique(self.src_space_labels, return_counts=True)
        points_per_region[region_ids] = counts
        return points_per_region


# MNE stuff
import mne


class SourceSpaceMNE(SourceSpaceData):

    def __init__(self, **kwargs):
        SourceSpaceData.__init__(self, **kwargs)

        self.use_cache = False
        self._src_space_path = op.join(self.cache_path, f"{self.src_space_string}.fif")

    @property  # cached_property
    def src_space(self):
        def compute_fn(self):
            src_space_pts = np.indices([256, 256, 256])[
                :, self.src_space_mask_lia
            ].T  # Transform src space mask to pc
            normals = np.repeat([[0, 0, 1]], len(src_space_pts), axis=0)

            # rr = point_cloud_to_voxel(src_space_pts)
            # rr = np.argwhere(rr != 0)
            # rr = mne.transforms.apply_trans(self.vox_mri_t, rr)

            pos = dict(rr=src_space_pts, nn=normals)
            src_space = mne.setup_volume_source_space(pos=pos, bem=self.bem)  # type: ignore
            return src_space
            print("Computing src space")
            # normals = np.repeat([[0, 0, 1]], len(self.src_space_points), axis=0)

            # rr = point_cloud_to_voxel(self.src_space_points)
            # rr = move_volume_from_ras_to_lia(rr)
            # # inv_aff = np.linalg.inv(self.affine)
            # # # Translation
            # # inv_aff[:, 3][2] = 132
            # # # Rotation
            # # inv_aff[:, 1][2] *= -1
            # # inv_aff[:, 2][1] *= -1
            # # inv_aff[:, 3][1] = -128
            # rr = apply_trans(self.affine, rr)
            # rr = np.argwhere(rr != 0)  # Back to point cloud
            # rr = mne.transforms.apply_trans(self.vox_mri_t, rr)
            # pos = dict(rr=rr, nn=normals)
            # src_space = mne.setup_volume_source_space(pos=pos)  # type: ignore (pos is float or dict)
            # return src_space
            normals = np.repeat([[0, 0, 1]], self.src_space_n_total_points, axis=0)

            rr = point_cloud_to_voxel(self.src_space_points)
            # rr = move_volume_from_ras_to_lia(rr)
            rr = np.argwhere(rr != 0)  # Back to point cloud
            # print(self.affine, self.t1_img.affine)
            # inv_aff = np.linalg.inv(self.t1_img.affine)
            # # Translation
            # inv_aff[:, 3][2] = 132
            # # Rotation
            # inv_aff[:, 1][2] *= -1
            # inv_aff[:, 2][1] *= -1
            # inv_aff[:, 3][1] = -128
            # rr = mne.transforms.apply_trans(self.affine, rr)
            # rr = rr / 1000
            print(f"rr shape: {rr.shape} normals shape: {normals.shape}")

            pos = dict(rr=rr, nn=normals)
            src_space = mne.setup_volume_source_space(pos=pos)  # type: ignore
            return src_space

        return (
            cache_mne_src(compute_fn, self._src_space_path, self)
            if self.use_cache
            else compute_fn(self)
        )
