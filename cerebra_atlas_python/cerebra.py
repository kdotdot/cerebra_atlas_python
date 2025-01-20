"""Main cerebra class
"""

import os.path as op
from appdirs import user_cache_dir
import numpy as np
from .data import CerebraData
from .data._transforms import (
    lia_points_to_ras_points,
    point_cloud_to_voxel,
    merge_voxel_grids,
)
from .plotting import Plotting
from .plotting.colors import normalize_colors_input
from .cerebra_mne import MNE
from typing import Dict


class CerebrA(CerebraData, Plotting, MNE):
    """CerebrA"""

    def __init__(self, **kwargs):
        # Path for processed files
        self.cache_path = op.join(user_cache_dir("cerebra_atlas_python"), "cerebra")

        CerebraData.__init__(self, cache_path=self.cache_path, **kwargs)
        Plotting.__init__(self, **kwargs)
        # SourceSpaceData should be initialized first
        MNE.__init__(self, cache_path=self.cache_path, cerebra_data=self, **kwargs)

    def get_brain_voxel_volume(self):
        """Returns a (256,256,256) np voxel array.
        Values ranging 0-103 represent region-ids from CerebrA

        Returns:
            np.ndarray: (256,256,256) numpy array.
        """
        return self.cerebra_volume

    def get_affine(self):
        """Returns affine matrix

        Returns:
            np.ndarray: Affine matrix.
        """
        return self.affine

    def get_metadata(self):
        """Returns a pandas DataFrame with region metadata

        Returns:
            pd.DataFrame: Region metadata
        """
        return self.cerebra_labels

    def get_brain_sparse(self) -> Dict[int, np.ndarray]:
        """Returns Dictionary containing sparse voxel grid for
        each region in RAS space[1-103]

        Returns:
            Dict[int, np.ndarray]: Dictionary with
            keys=region_id and
            values=[[x,y,z]...] points belonging to each region
        """
        return self.cerebra_sparse

    def get_src_space_points(self):
        """Returns a np array of src space points in RAS space.
        Values ranging 0-64 represent cortical ids

        Returns:
            np.ndarray: source space points (RAS) numpy array.
        """
        return self.src_space_points

    def get_src_space_labels(self):
        """Returns a np array of len(src_space_points).
        Values ranging 0-64 represent cortical ids

        Returns:
            np.ndarray: source space labels numpy array.
        """
        return self.src_space_labels

    def plot2d(self, **kwargs):
        """Plot 2D brain"""
        self._plot(kind="2d", **kwargs)

    def orthoview(self, **kwargs):
        """Plot 2D brain with orthoview"""
        self._plot(kind="orthoview", **kwargs)

    def plot3d(self, rotate_mode=1, save_path=None, update_fn=None, **kwargs):
        """Plot 3D brain"""
        plot_data_ = {
            "rotate_mode": rotate_mode,
            "save_path": save_path,
            "update_fn": update_fn,
        }
        self._plot(kind="3d", plot_data_=plot_data_, **kwargs)

    def _plot(self, colors=None, plot_data_=None, **kwargs):
        # Prepare plot data
        plot_data = {
            "affine": self.affine,
            "cerebra_volume": self.cerebra_volume,
            "src_space_points": self.src_space_points,
            "src_space_labels": self.src_space_labels,
            "bem_vertices_vox_ras": self._get_bem_vertices_vox_ras(),
            "bem_normals_vox_ras": self._get_bem_normals_vox_ras(),
            "bem_triangles": self.get_bem_triangles(),
            "info": (
                self.info
                if (self.montage_name is not None and self.head_size is not None)
                else None
            ),
            "fiducials": self.fiducials,
        }
        if plot_data_ is not None:
            plot_data = {**plot_data, **plot_data_}
        plot_data["colors"] = normalize_colors_input(self.src_space_labels, colors)
        self._plot_data(plot_data=plot_data, **kwargs)

    def _get_bem_vertices_vox_lia(self):
        return np.array(
            [self.apply_mri_vox_t(layer) for layer in self.get_bem_vertices_mri()]
        )

    def _get_bem_normals_vox_lia(self):
        return np.array(
            [self.apply_mri_vox_t(layer) for layer in self.get_bem_normals_mri()]
        )

    def _get_bem_vertices_vox_ras(self):
        return np.array(
            [
                lia_points_to_ras_points(layer)
                for layer in self._get_bem_vertices_vox_lia()
            ]
        )

    def _get_bem_normals_vox_ras(self):
        return np.array(
            [
                lia_points_to_ras_points(layer)
                for layer in self._get_bem_normals_vox_lia()
            ]
        )

    def _get_bem_volume_ras(self, include_layers=[0, 1, 2]) -> np.ndarray:
        for i, layer_pts in enumerate(self.get_bem_vertices_mri()[include_layers]):
            layer_pts = self.apply_mri_vox_t(layer_pts)
            layer_pts = lia_points_to_ras_points(layer_pts)
            layer_vol = point_cloud_to_voxel(layer_pts, vox_value=i + 1)
            if i == 0:
                volume = layer_vol
            else:
                volume = merge_voxel_grids(volume, layer_vol)  # type: ignore
        return volume  # type: ignore


if __name__ == "__main__":
    cerebra = CerebrA()
    cerebra.corregistration()
