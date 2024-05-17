#!/usr/bin/env python
"""
SourceSpace
"""
import os.path as op
from functools import cached_property
import appdirs
import numpy as np
from ._cache import cache_np
from ._transforms import volume_lia_to_ras
from .cerebra_data import CerebraData

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
class SourceSpaceData(CerebraData):
    """Uses CerebraData to generate the source space mask and points"""

    def __init__(
        self,
        data_path=None,
        cache_path=None,
        source_space_grid_size: int = 3,
        source_space_include_wm: bool = False,
        source_space_include_non_cortical: bool = False,
        **kwargs,
    ):
        """_summary_

        Args:
            data_path (_type_, optional): Path to cerebra data dir. Defaults to None.
            cache_path (_type_, optional): Path to cerebra cache dir. Defaults to None.
            source_space_grid_size (int, optional): Grid size for generating the source space, 
                bigger means more downsampling (less src space points). Defaults to 3.
            source_space_include_wm (bool, optional): Whether to include whitematter 
                in the source space. Defaults to False.
            source_space_include_non_cortical (bool, optional):Whether to include non-cortical 
                regions in the source space. Defaults to False.
            kwargs: Additional arguments to pass to CerebraData
        """
        self.cerebra_data_path = (
            op.dirname(__file__) + "/cerebra_data" if data_path is None else data_path
        )
        self.cache_path_cerebra: str = (
            op.join(appdirs.user_cache_dir("cerebra_atlas_python"), "cerebra")
            if cache_path is None
            else cache_path
        )

        CerebraData.__init__(
            self,
            data_path=self.cerebra_data_path,
            cache_path=self.cache_path_cerebra,
            **kwargs,
        )

        self.source_space_grid_size: int = source_space_grid_size
        self.source_space_include_wm: bool = source_space_include_wm
        self.source_space_include_non_cortical: bool = source_space_include_non_cortical

        wm_str = "wm" if self.source_space_include_wm else ""
        nc_str = "_nc" if self.source_space_include_non_cortical else ""
        self.src_space_string = f"src_space_{self.source_space_grid_size}mm{wm_str}{nc_str}"
        self._src_space_mask_path = op.join(
            self.cache_path_cerebra, f"{self.src_space_string}_mask.npy"
        )
        self._src_space_mask_lia_path = op.join(
            self.cache_path_cerebra, f"{self.src_space_string}_lia_mask.npy"
        )
        self._src_space_points_path = op.join(
            self.cache_path_cerebra, f"{self.src_space_string}_src_pts.npy"
        )
        self._src_space_points_path_lia = op.join(
            self.cache_path_cerebra, f"{self.src_space_string}_src_pts_lia.npy"
        )

    @cached_property
    def src_space_mask(self) -> np.ndarray:
        """Mask for cerebra_volume in RAS space.
        Indicates whether point from (256,256,256) belongs to the source space

        Returns:
            np.ndarray: (256,256,256) mask in RAS space
        """
        return cache_np(
            volume_lia_to_ras, self._src_space_mask_path, self.src_space_mask_lia
        )

    @cached_property
    def src_space_mask_lia(self) -> np.ndarray:
        """Mask for cerebra_volume in LIA space.
        Indicates whether point from (256,256,256) belongs to the source space

        Returns:
            np.ndarray: (256,256,256) mask in LIA space
        """
        return cache_np(self.get_source_space_mask, self._src_space_mask_lia_path)

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
        """Array of length N x 3 containing N points in LIA space

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
        return self.cerebra_volume[self.src_space_mask]

    @cached_property
    def src_space_n_points_per_region(self) -> np.ndarray:
        """Array of length 104 containing number of points in each region

        Returns:
            np.ndarray: 1D array with number of points in each region
        """
        points_per_region = np.zeros(len(self.region_ids))
        region_ids, counts = np.unique(self.src_space_labels, return_counts=True)
        points_per_region[region_ids] = counts
        return points_per_region

    @cached_property
    def src_space_n_total_points(self) -> int:
        """
        Returns:
            int: The total number of points in the source space (N).
        """
        return len(self.src_space_points)

    # pylint: disable=too-many-locals
    def get_source_space_mask(self, coord_frame="ras") -> np.ndarray:
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

        if coord_frame == "lia":
            volume, _ = self._get_wm_filled_cerebra_volume_aff_lia()
        elif coord_frame == "ras":
            volume = self.cerebra_volume
        else:
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
            not_zero_mask = volume != self.get_region_id_from_region_name("Empty")
            not_wm_mask = volume != self.get_region_id_from_region_name("White matter")
            downsampled_not_zero_mask = np.logical_and(grid_mask, not_zero_mask)
            downsampled_not_wm_mask = np.logical_and(grid_mask, not_wm_mask)
            # Combined mask is not zero and not wm
            combined_mask = np.logical_and(
                downsampled_not_zero_mask, downsampled_not_wm_mask
            )

        else:  # Keep only cortical:
            # Get cortical region ids
            cortical_ids = self.get_cortical_region_ids()
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
            whitematter_mask = volume == self.get_region_id_from_region_name(
                "White matter"
            )
            # Downsample
            downsampled_whitematter_mask = np.logical_and(grid_mask, whitematter_mask)
            combined_mask = np.logical_or(combined_mask, downsampled_whitematter_mask)

        return combined_mask
