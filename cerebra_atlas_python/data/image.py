#!/usr/bin/env python
"""
This module handles stuff related with data/cerebra_data/CerebrA_in_t1.mgz
"""

from typing import Tuple, cast
import os.path as op
import numpy as np
from nibabel.loadsave import load
from nibabel.nifti1 import Nifti1Image

from ._transforms import read_mri_info, lia_to_ras, apply_trans, apply_inverse_trans


class Image:
    """Handles CerebrA_in_t1.mgz image related operations

    Attributes:
        _cerebra_img (nibabel.nifti1.Nifti1Image): Nibabel image object for CerebrA_in_t1.mgz.
        _cerebra_img_path (str): Path to the CerebrA_in_t1.mgz file.
        vox_ras_t (mne.transforms.Transform): Transform from voxel to RAS (non-zero origin) space.
        vox_mri_t (mne.transforms.Transform): Transform from voxel to MRI space.
        mri_ras_t (mne.transforms.Transform): Transform from MRI to RAS (non-zero origin) space.
    """

    def __init__(
        self, cerebra_data_path: str, image_name: str = "CerebrA_in_t1.mgz", **kwargs
    ):
        """
        Initialize the Image class.

        Args:
            cerebra_data_path (str): Path to the directory containing the CerebrA_in_t1.mgz file.
            image_name (str, optional): Name of the image file. Defaults to "CerebrA_in_t1.mgz".
        """
        self._cerebra_img: Nifti1Image | None = None
        self._cerebra_img_path = op.join(cerebra_data_path, image_name)
        self.vox_ras_t, self.vox_mri_t, self.mri_ras_t = read_mri_info(
            self._cerebra_img_path
        )

    @property
    def cerebra_img(self) -> Nifti1Image:
        """
        Nibabel image CerebrA_in_t1.mgz.

        Returns:
            nibabel.nifti1.Nifti1Image: Nibabel image object for CerebrA_in_t1.mgz.
        """
        if self._cerebra_img is None:
            self._cerebra_img = cast(Nifti1Image, load(self._cerebra_img_path))
        return self._cerebra_img

    def get_cerebra_vox_affine_ras(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the cerebra volume (256, 256, 256) and affine in RAS space.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - np.ndarray: Cerebra volume in RAS space.
                - np.ndarray: Affine matrix for the cerebra volume in RAS space.
        """
        return lia_to_ras(self.cerebra_img.get_fdata(), self.cerebra_img.affine)

    def get_cerebra_vox_affine_lia(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the cerebra volume (256, 256, 256) and affine in LIA space.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - np.ndarray: Cerebra volume in LIA space.
                - np.ndarray: Affine matrix for the cerebra volume in LIA space.
        """
        return self.cerebra_img.get_fdata(), self.cerebra_img.affine

    def apply_vox_ras_t(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the vox_ras transformation to data.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        return apply_trans(self.vox_ras_t["trans"], data)

    def apply_ras_vox_t(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the inverse vox_ras transformation to data, transforming it to voxel space (int).

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data in voxel space (int).
        """
        return apply_inverse_trans(self.vox_ras_t["trans"], data).astype(int)

    def apply_vox_mri_t(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the vox_mri transformation to data.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        return apply_trans(self.vox_mri_t["trans"], data)

    def apply_mri_vox_t(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the inverse vox_mri transformation to data, transforming it to voxel space (int).

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data in voxel space (int).
        """
        return apply_inverse_trans(self.vox_mri_t["trans"], data).astype(int)

    def apply_mri_ras_t(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the mri_ras transformation to data.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        return apply_trans(self.mri_ras_t["trans"], data)

    def apply_ras_mri_t(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the inverse mri_ras transformation to data.

        Args:
            data (np.ndarray): Input data.

        Returns:
            np.ndarray: Transformed data.
        """
        return apply_inverse_trans(self.mri_ras_t["trans"], data)
