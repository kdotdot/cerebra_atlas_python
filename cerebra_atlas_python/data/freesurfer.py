#!/usr/bin/env python
"""
This module handles stuff related with data/cerebra_data/FreeSurfer
"""

import os.path as op
from typing import Tuple
import nibabel as nib  # NOTE: nib typing does not work
import mne
import numpy as np
from ._transforms import lia_to_ras, apply_trans, apply_inverse_trans


class FreeSurfer:
    """Handles FreeSurfer related operations

    Attributes:
        subjects_dir (str): Freesurfer subjects dir
        subject_name (str): Subject name
        icbm152_dir (str): ICBM152 subject dir
    """

    def __init__(self, cerebra_data_path: str, **kwargs):
        """Handles FreeSurfer related operations"""

        self.subjects_dir = op.join(cerebra_data_path, "FreeSurfer/subjects")
        self.subject_name = "icbm152"
        self.icbm152_dir = op.join(self.subjects_dir, self.subject_name)

        self._t1_img = None
        self._wm_img = None

        self._t1_img_path = op.join(self.icbm152_dir, "mri/T1.mgz")
        self._wm_img_path = op.join(self.icbm152_dir, "mri/wm.mgz")

        self.fiducials, _ = mne.io.read_fiducials(
            op.join(self.icbm152_dir, "bem/icbm152-fiducials.fif")
        )
        # self.head_mri_trans = mne.read_trans(
        #     op.join(self.icbm152_dir, "bem/head_mri_t.fif")
        # )

    @property
    def t1_img(self):
        """Nibabel image T1.mgz"""
        if self._t1_img is None:
            self._t1_img = nib.load(self._t1_img_path)  # type: ignore
        return self._t1_img

    @property
    def wm_img(self):
        """Nibabel image wm.mgz"""
        if self._wm_img is None:
            self._wm_img = nib.load(self._wm_img_path)  # type: ignore
        return self._wm_img

    def get_t1_vox_affine_ras(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get t1 volume (256,256,256) and affine in RAS space"""
        return lia_to_ras(self.t1_img.get_fdata(), self.t1_img.affine)  # type: ignore

    def get_t1_vox_affine_lia(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get t1 volume (256,256,256) and affine in LIA space"""
        return self.t1_img.get_fdata(), self.t1_img.affine  # type: ignore

    def get_wm_vox_affine_ras(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get wm volume (256,256,256) and affine in RAS space"""
        return lia_to_ras(self.wm_img.get_fdata(), self.wm_img.affine)  # type: ignore

    def get_wm_vox_affine_lia(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get wm volume (256,256,256) and affine in LIA space"""
        return self.wm_img.get_fdata(), self.wm_img.affine  # type: ignore

    # def apply_head_mri_t(self, points):
    #     """Apply head-mri transformation"""
    #     return apply_trans(data=points, trans=self.head_mri_trans)  # type: ignore

    # def apply_mri_head_t(self, points):
    #     """Apply inverse head-mri transformation"""
    #     return apply_inverse_trans(data=points, trans=self.head_mri_trans)  # type: ignore
