"""
Handles Freesurfer related operations
"""

import os.path as op
import logging
import mne
import nibabel as nib
import numpy as np
import appdirs
from .transforms import lia_to_ras, read_mri_info, lia_points_to_ras_points
from .icbm152_bem import ICBM152BEM
from .utils import point_cloud_to_voxel, merge_voxel_grids

logger = logging.getLogger(__name__)


class CerebraBase(ICBM152BEM):
    def __init__(self, **kwargs):

        ICBM152BEM.__init__(self, **kwargs)

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
                volume = merge_voxel_grids(volume, layer_vol)
        return volume
