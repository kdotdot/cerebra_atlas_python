"""
Handles Freesurfer related operations
"""
import os.path as op
import logging
import mne
import nibabel as nib
import numpy as np
import appdirs
from .config import Config
from .transforms import lia_to_ras, read_mri_info
from .icbm152_bem import ICBM152BEM
logger = logging.getLogger(__name__)

class CerebraBase(ICBM152BEM,Config):
    def __init__(self, **kwargs):
        self.cerebra_data_path: str = op.dirname(__file__) + "/cerebra_data"
        self.cache_path_cerebra: str = op.join(appdirs.user_cache_dir("cerebra_atlas_python"), "cerebra")
        Config.__init__(
            self,
            class_name=self.__class__.__name__,
            **kwargs,
        )
        ICBM152BEM.__init__(self, **kwargs)

        self._t1_img = None
        self._wm_img = None
        self._cerebra_img = None

        self._t1_img_path = op.join(self.cerebra_data_path, "FreeSurfer/subjects/icbm152/mri/T1.mgz")
        self._wm_img_path = op.join(self.cerebra_data_path, "FreeSurfer/subjects/icbm152/mri/wm.mgz")
        self._cerebra_img_path = op.join(self.cerebra_data_path, "CerebrA_in_t1.mgz")

        self.vox_ras_t, self.vox_mri_t, self.mri_ras_t,_,_ = read_mri_info(self._cerebra_img_path)

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
    
    def get_t1_volume_ras(self):
        return lia_to_ras(self.t1_img.get_fdata(), self.t1_img.affine)
    
    def get_wm_volume_ras(self):
        return lia_to_ras(self.wm_img.get_fdata(), self.wm_img.affine)
    
    def get_cerebra_volume_ras(self):
        return lia_to_ras(self.cerebra_img.get_fdata(), self.cerebra_img.affine)
    
    def get_bem_vertices_vox(self):
        return mne.transforms.apply_trans(np.linalg.inv(self.vox_mri_t["trans"]), self.get_bem_vertices_mri()).astype(int)