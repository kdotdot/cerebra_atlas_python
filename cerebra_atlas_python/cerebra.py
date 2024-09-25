"""Main cerebra class
"""

import os.path as op
import appdirs
import numpy as np
from .data import CerebraData
from .data._transforms import (
    lia_points_to_ras_points,
    point_cloud_to_voxel,
    merge_voxel_grids,
)
from .plotting import Plotting
from .cerebra_mne import MNE


class CerebrA(CerebraData, Plotting, MNE):
    """Main cerebra class SA"""

    def __init__(self, **kwargs):

        print(kwargs)

        self.cache_path = op.join(
            appdirs.user_cache_dir("cerebra_atlas_python"), "cerebra"
        )

        CerebraData.__init__(self, cache_path=self.cache_path, **kwargs)
        Plotting.__init__(self, **kwargs)
        # SourceSpaceData should be initialized first
        MNE.__init__(self, cache_path=self.cache_path, cerebra_data=self, **kwargs)

    def corregistration(self, **kwargs):
        """Manually generate fiducials.fif and head-mri-trans.fif
        Saved to standard location (cerebra_data/FreeSurfer/bem/)
        """
        assert (
            "montage" in kwargs or "montage_name" in kwargs
        ), "Either MME montage or montage_name should be provided for corregistration"
        self._corregistration(**kwargs)

    def _prepare_plot_data(self, plot_data_=None):
        plot_data = {
            "affine": self.affine,
            "cerebra_volume": self.cerebra_volume,
            "src_space_points": self.src_space_points,
            "src_space_labels": self.src_space_labels,
            "bem_vertices_vox_ras": self.get_bem_vertices_vox_ras(),
            "bem_normals_vox_ras": self.get_bem_normals_vox_ras(),
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
        return plot_data

    def _plot(self, colors=None, plot_data=None, **kwargs):
        plot_data = self._prepare_plot_data(plot_data)
        if colors is not None:
            plot_data["colors"] = colors
        self._plot_data(plot_data=plot_data, **kwargs)

    def orthoview(self, **kwargs):
        """Plot 2D brain with orthoview"""
        self._plot(kind="orthoview", **kwargs)

    def plot2d(self, **kwargs):
        """Plot 2D brain"""
        self._plot(kind="2d", **kwargs)

    def plot3d(self, rotate_mode=1, save_path=None, **kwargs):
        """Plot 3D brain"""
        plot_data = {"rotate_mode": rotate_mode, "save_path": save_path}
        self._plot(kind="3d", plot_data=plot_data, **kwargs)

    def get_bem_vertices_vox_lia(self):
        return np.array(
            [self.apply_mri_vox_t(layer) for layer in self.get_bem_vertices_mri()]
        )

    def get_bem_normals_vox_lia(self):
        return np.array(
            [self.apply_mri_vox_t(layer) for layer in self.get_bem_normals_mri()]
        )

    def get_bem_vertices_vox_ras(self):
        return np.array(
            [
                lia_points_to_ras_points(layer)
                for layer in self.get_bem_vertices_vox_lia()
            ]
        )

    def get_bem_normals_vox_ras(self):
        return np.array(
            [
                lia_points_to_ras_points(layer)
                for layer in self.get_bem_normals_vox_lia()
            ]
        )

    def get_bem_volume_ras(self, include_layers=[0, 1, 2]) -> np.ndarray:
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
