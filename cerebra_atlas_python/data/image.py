#!/usr/bin/env python
"""
This module handles stuff related with data/cerebra_data/CerebrA_in_t1.mgz
"""

from typing import Tuple
import os.path as op
import numpy as np
from nibabel.loadsave import load
from ._transforms import read_mri_info, lia_to_ras, apply_trans, apply_inverse_trans


class Image:
    """Handles CerebrA_in_t1.mgz image related operations"""

    def __init__(self, cerebra_data_path: str, image_name="CerebrA_in_t1.mgz"):
        self._cerebra_img = None
        self._cerebra_img_path = op.join(cerebra_data_path, image_name)
        self.vox_ras_t, self.vox_mri_t, self.mri_ras_t = read_mri_info(
            self._cerebra_img_path
        )

    @property
    def cerebra_img(self):
        """Nibabel image CerebrA_in_t1.mgz"""
        if self._cerebra_img is None:
            self._cerebra_img = load(self._cerebra_img_path)
        return self._cerebra_img

    def get_cerebra_vox_affine_ras(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get cerebra volume (256,256,256) and affine in RAS space"""
        return lia_to_ras(self.cerebra_img.get_fdata(), self.cerebra_img.affine)  # type: ignore

    def get_cerebra_vox_affine_lia(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get cerebra volume (256,256,256) and affine in LIA space"""
        return self.cerebra_img.get_fdata(), self.cerebra_img.affine  # type: ignore

    def apply_vox_ras_t(self, data: np.ndarray) -> np.ndarray:
        """Apply vox_ras transformation"""
        return apply_trans(self.vox_ras_t["trans"], data)

    def apply_ras_vox_t(self, data: np.ndarray) -> np.ndarray:
        """Apply inverse vox_ras transformation, transform to voxel space (int)"""
        return apply_inverse_trans(self.vox_ras_t["trans"], data).astype(int)

    def apply_vox_mri_t(self, data: np.ndarray) -> np.ndarray:
        """Apply vox_mri transformation"""
        return apply_trans(self.vox_mri_t["trans"], data)

    def apply_mri_vox_t(self, data: np.ndarray) -> np.ndarray:
        """Apply inverse vox_mri transformation, transform to voxel space (int)"""
        return apply_inverse_trans(self.vox_mri_t["trans"], data).astype(int)

    def apply_mri_ras_t(self, data: np.ndarray) -> np.ndarray:
        """Apply mri_ras transformation"""
        return apply_trans(self.mri_ras_t["trans"], data)

    def apply_ras_mri_t(self, data: np.ndarray) -> np.ndarray:
        """Apply inverse mri_ras transformation"""
        return apply_inverse_trans(self.mri_ras_t["trans"], data)
