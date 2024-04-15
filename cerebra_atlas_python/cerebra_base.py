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
        self.cerebra_data_path: str = op.dirname(__file__) + "/cerebra_data"
        self.cache_path_cerebra: str = op.join(appdirs.user_cache_dir("cerebra_atlas_python"), "cerebra")
        ICBM152BEM.__init__(self, **kwargs)

        self._t1_img = None
        self._wm_img = None
        self._cerebra_img = None

        self._t1_img_path = op.join(self.cerebra_data_path, "FreeSurfer/subjects/icbm152/mri/T1.mgz")
        self._wm_img_path = op.join(self.cerebra_data_path, "FreeSurfer/subjects/icbm152/mri/wm.mgz")
        self._cerebra_img_path = op.join(self.cerebra_data_path, "CerebrA_in_t1.mgz")

        self.vox_ras_t, self.vox_mri_t, self.mri_ras_t,_,_ = read_mri_info(self._cerebra_img_path)

        self.bem_colors = [[0,0.1,1],[0.1,0.2,0.9],[0.2,0.1,0.95]]
        self.cortical_color = [0.3,1,0.5]
        self.non_cortical_color = [1,0.4,0.3]

        self.fiducials,_ = mne.io.read_fiducials(op.join(self.subject_dir, "bem/icbm152-fiducials.fif"))
        self.head_mri_trans = mne.read_trans(op.join(self.subject_dir, "bem/head_mri_t.fif"))

    @property
    def t1_img(self):
        if self._t1_img is None:
            self._t1_img = nib.load(self._t1_img_path)
        return self._t1_img
    
    @property
    def wm_img(self):
        if self._wm_img is None:
            self._wm_img = nib.load(self._wm_img_path)
        return self._wm_img
    
    @property
    def cerebra_img(self):
        if self._cerebra_img is None:
            self._cerebra_img = nib.load(self._cerebra_img_path)
        return self._cerebra_img
    
    def apply_vox_ras_t(self, points):
        return mne.transforms.apply_trans(self.vox_ras_t["trans"], points)
    def apply_ras_vox_t(self, points):
        return mne.transforms.apply_trans(np.linalg.inv(self.vox_ras_t["trans"]), points).astype(int)
    def apply_vox_mri_t(self, points):
        return mne.transforms.apply_trans(self.vox_mri_t["trans"], points)
    def apply_mri_vox_t(self, points):
        return mne.transforms.apply_trans(np.linalg.inv(self.vox_mri_t["trans"]), points).astype(int)
    def apply_mri_ras_t(self, points):
        return mne.transforms.apply_trans(self.mri_ras_t["trans"], points)
    def apply_ras_mri_t(self, points):
        return mne.transforms.apply_trans(np.linalg.inv(self.mri_ras_t["trans"]),points)
    def apply_head_mri_t(self, points):
        return mne.transforms.apply_trans(self.head_mri_trans,points)
    def apply_mri_head_t(self, points):
        return mne.transforms.apply_trans(np.linalg.inv(self.head_mri_trans["trans"]),points)
    
    def get_t1_vox_affine_ras(self):
        return lia_to_ras(self.t1_img.get_fdata(), self.t1_img.affine)
    
    def get_wm_vox_affine_ras(self):
        return lia_to_ras(self.wm_img.get_fdata(), self.wm_img.affine)
    
    def get_cerebra_vox_affine_ras(self):
        return lia_to_ras(self.cerebra_img.get_fdata(), self.cerebra_img.affine)
    
    def get_bem_vertices_vox_lia(self):
        return np.array([self.apply_mri_vox_t(layer) for layer in self.get_bem_vertices_mri()]) 
    
    def get_bem_normals_vox_lia(self):
        return np.array([self.apply_mri_vox_t(layer) for layer in self.get_bem_normals_mri()]) 
    
    def get_bem_vertices_vox_ras(self):
        return np.array([lia_points_to_ras_points(layer) for layer in self.get_bem_vertices_vox_lia()]) 
    
    def get_bem_normals_vox_ras(self):
        return np.array([lia_points_to_ras_points(layer) for layer in self.get_bem_normals_vox_lia()]) 
    
    def get_bem_volume_ras(self, include_layers = [0,1,2])-> np.ndarray:
        for i, layer_pts in enumerate(self.get_bem_vertices_mri()[include_layers]):
            layer_pts = self.apply_mri_vox_t(layer_pts)
            layer_pts = lia_points_to_ras_points(layer_pts)
            layer_vol = point_cloud_to_voxel(layer_pts, vox_value=i+1)
            if i == 0:
                volume = layer_vol
            else:
                volume = merge_voxel_grids(volume, layer_vol)
        return volume
