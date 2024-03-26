"""
MNIAverage
"""
import os
import os.path as op
import logging
from typing import Tuple, Optional, List

import mne
import nibabel as nib
import numpy as np

from core.config import Config


class MNIAverage(Config):
    def __init__(self, config_path=None, **kwargs):
        self.mniaverage_output_path: str = "./generated/mni_average"
        self.bem_conductivity: Tuple[float, float, float] = (0.33, 0.0042, 0.33)
        self.bem_ico: int = 4
        self.cerebra_data_path: str = op.dirname(__file__) + "/cerebra_data"
       
        super().__init__(
            config_path=config_path,
            class_name=self.__class__.__name__,
            **kwargs,
        )

        # Always load (fast/required)
        self.fiducials: Optional[List[mne.io._digitization.DigPoint]] = None
        self.head_mri_t: Optional[mne.Transform] = None
        self.mri_ras_t: Optional[mne.Transform] = None

        # Load on demand [using @property] (slow/rarely used)
        self._bem: Optional[mne.bem.ConductorModel] = None
        self._t1 = None
        self._wm = None
        self._info = None

        # If output folder does not exist, create it
        if not op.exists(self.mniaverage_output_path):
            os.makedirs(self.mniaverage_output_path, exist_ok=True)

        # Input paths
        self.subject_name = "icbm152"
        self.subjects_dir = op.join(self.cerebra_data_path, "subjects")
        self.subject_dir = op.join(self.subjects_dir, self.subject_name)
        self.bem_folder_path = op.join(self.subject_dir, "bem")
        self.fiducials_path = op.join(
            self.bem_folder_path, f"{self.subject_name}-fiducials.fif"
        )
        self.wm_path = op.join(self.subject_dir, "mri/wm.mgz")
        self.t1_path = op.join(self.subject_dir, "mri/T1.mgz")
        self.head_mri_t_path = op.join(self.subject_dir, "head_mri_t.fif")
        self.info_path = op.join(self.cerebra_data_path, "info.fif")
        self.mri_ras_t_path = op.join(self.cerebra_data_path, "mri_ras-trans.fif")

        # Output paths
        self._bem_solution_path = op.join(
            self.mniaverage_output_path,
            f"{self.name}.fif",
        )

        # Mapping for making sure bem indices refer to the same surface every time
        self.bem_names = {1: "outer_skin", 2: "outer_skull", 3: "inner_skull"}

        self._set_fiducials()
        self._set_head_mri_t()
        self._set_mri_ras_t()

    # * PROPERTIES
    @property
    def bem_conductivity_string(self) -> str:
        """
        Property to get the BEM conductivity values as a formatted string.
        Returns:
            str: A string r_bemepresenting the BEM conductivity values, formatted as 'bem_value1_value2_value3'.
        """
        return "bem_" + "".join([str(x) + "_" for x in self.bem_conductivity])[:-1]

    @property
    def name(self) -> str:
        """Property to get the enhanced name of the class instance.
        Returns:
            str: Enhanced name of the class instance, including BEM conductivity and icosahedron level.
        """
        return f"{self.bem_conductivity_string}_ico_{self.bem_ico}"

    @property
    def bem(self):
        if self._bem is None:
            self._set_bem()
        return self._bem

    @property
    def t1(self):
        if self._t1 is None:
            self._t1 = nib.load(self.t1_path)
        return self._t1

    @property
    def wm(self):
        if self._wm is None:
            self._wm = nib.load(self.wm_path)
        return self._wm

    @property
    def info(self):
        if self._info is None:
            self._info = mne.io.read_info(self.info_path)
        return self._info

    # * SETTERS
    def _set_bem(self):
        """Internal method to set up the boundary element model (BEM)."""
        if not op.exists(self._bem_solution_path):
            logging.info("Generating boundary element model... | %s", self.name)
            model = mne.make_bem_model(
                subject=self.subject_name,
                ico=self.bem_ico,
                conductivity=self.bem_conductivity,
                subjects_dir=self.subjects_dir,
            )
            self._bem = mne.make_bem_solution(model)
            mne.write_bem_solution(
                self._bem_solution_path, self.bem, overwrite=True, verbose=True
            )
        else:
            logging.info("Loading boundary element model from disk | %s", self.name)
            self._bem = mne.read_bem_solution(self._bem_solution_path, verbose=False)

    def _set_fiducials(self):
        """Internal method to read manually aligned fiducials."""
        self.fiducials, _coordinate_frame = mne.io.read_fiducials(self.fiducials_path)

    def _set_head_mri_t(self):
        """Internal method to read manually aligned fiducials."""
        self.head_mri_t = mne.read_trans(self.head_mri_t_path)

    def _set_mri_ras_t(self):
        # MRI (surface RAS)->RAS (non-zero origin)
        self.mri_ras_t = mne.read_trans(self.mri_ras_t_path)

    # * TRANSFORMS
    def mri_to_ras_nzo(self, pts: np.ndarray) -> np.ndarray:
        """
        Converts points from MRI (surface RAS) coordinates to RAS coordinates with non-zero origin.
        Args:
            pts (np.ndarray): Points in MRI (surface RAS) coordinates.
        Returns:
            np.ndarray: Points converted to RAS (non-zero origin) coordinates.
        """
        res = mne.transforms.apply_trans(self.mri_ras_t["trans"], pts) * 1000
        return res

    def ras_nzo_to_mri(self, pts: np.ndarray) -> np.ndarray:
        """
        # TODO: Update docstring
        Converts points from MRI (surface RAS) coordinates to RAS coordinates with non-zero origin.
        Args:
            pts (np.ndarray): Points in MRI (surface RAS) coordinates.
        Returns:
            np.ndarray: Points converted to RAS (non-zero origin) coordinates.
        """
        # print(np.linalg.inv(self.src["mri_ras_t"]), pts)
        return (
            mne.transforms.apply_trans(np.linalg.inv(self.mri_ras_t["trans"]), pts)
            / 1000
        )

    def apply_head_mri_t(self, pts: np.ndarray) -> np.ndarray:
        return mne.transforms.apply_trans(self.head_mri_t, pts) / 1000

    # * METHODS
    def get_bem_surfaces_ras_nzo(
        self, transform: Optional[mne.Transform] = None
    ) -> np.ndarray:
        """
        Retrieves the BEM surfaces in RAS coordinates with non-zero origin, with an optional transformation.

        Args:
            transform (Optional[mne.Transform]): An optional transformation to apply to the BEM surfaces.

        Returns:
            np.ndarray: An array of BEM surfaces in RAS (non-zero origin) coordinates.
        """
        surfaces = []
        for surf in self.bem["surfs"]:
            surf = surf["rr"]
            bem_surface = self.mri_to_ras_nzo(
                surf
            )  # Move volume from MRI space to RAS (non-zero origin)
            if transform is not None:
                bem_surface = mne.transforms.apply_trans(transform, bem_surface).astype(
                    int
                )
            surfaces.append(bem_surface)
        surfaces = np.array(surfaces).astype(int)
        return surfaces


if __name__ == "__main__":
    mniAverage = MNIAverage(bem_ico=4)
