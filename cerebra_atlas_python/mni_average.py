"""
This module provides classes and functions for configuring and managing
MNIAverage surfaces and related data processing operations.
It includes functionality for handling boundary element models (BEM),
volume source spaces, and fiducial points, tailored for MNE-Python usage.
"""
import os
import os.path as op
import logging
from typing import Tuple, Optional, Dict, List

import mne
import nibabel as nib
import numpy as np
import pandas as pd

from cerebra_atlas_python.config import BaseConfig
from cerebra_atlas_python.plotting import get_cmap_colors_hex


class MNIAverage(BaseConfig):
    def __init__(self, config_path=op.dirname(__file__) + "/config.ini", **kwargs):
        self.mniaverage_output_path: str = None
        self.bem_conductivity: Tuple[float, float, float] = None
        self.bem_ico: int = None
        self.cerebra_data_path: str = None
        default_config = {
            "mniaverage_output_path": "./mni_average",
            "bem_conductivity": (0.33, 0.0042, 0.33),
            "bem_ico": 4,
            "cerebra_data_path": op.dirname(__file__) + "/cerebra_data",
        }

        super().__init__(
            config_path=config_path,
            parent_name=self.__class__.__name__,
            default_config=default_config,
            **kwargs,
        )

        self.bem: Optional[mne.bem.ConductorModel] = None
        self.fiducials: Optional[List[mne.io._digitization.DigPoint]] = None

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

        # Output paths
        self._bem_solution_path = op.join(
            self.mniaverage_output_path,
            f"{self.name}.fif",
        )

        # Mapping for making sure bem indices refer to the same surface every time
        self.bem_names = {1: "outer_skin", 2: "outer_skull", 3: "inner_skull"}

        self._set_bem()
        self._set_fiducials()
        self._set_head_mri_t()
        self._set_info()
        self._set_wm()
        self._set_t1()

        src_space = mne.setup_volume_source_space(
            subject=self.subject_name
        )  # NOTE: Remove from here. Put in 0.0 notebook
        self.mri_ras_t = src_space[0][
            "mri_ras_t"
        ]  # MRI (surface RAS)->RAS (non-zero origin)
        self.src_mri_t = src_space[0]["src_mri_t"]  # MRI voxel->MRI (surface RAS)
        self.vox_mri_t = src_space[0]["vox_mri_t"]  # MRI voxel->MRI (surface RAS)

    @property
    def bem_conductivity_string(self) -> str:
        """
        Property to get the BEM conductivity values as a formatted string.
        Returns:
            str: A string representing the BEM conductivity values, formatted as 'bem_value1_value2_value3'.
        """
        return "bem_" + "".join([str(x) + "_" for x in self.bem_conductivity])[:-1]

    @property
    def name(self) -> str:
        """Property to get the enhanced name of the class instance.
        Returns:
            str: Enhanced name of the class instance, including BEM conductivity and icosahedron level.
        """
        return f"{super().name}_{self.bem_conductivity_string}_ico_{self.bem_ico}"

    # @property
    # def src(self) -> Dict:
    #     """
    #     This property provides convenient access to the first source space object, assuming
    #     the volume source space has been set.

    #     Returns:
    #         The first source space object, if available.
    #     """
    #     return self.vol_src[0]

    # # If vol src fif does not exist, create it, otherwise read it
    # # NOTE: UNUSED (remove)
    # def _set_vol_src(self):
    #     """
    #     Internal method to set up the volume source space.

    #     This method checks if the volume source space file exists. If not, it generates a new
    #     volume source space and saves it to the specified path. If the file exists, it reads the
    #     source space from the disk.

    #     Uses the 'inner_skull.surf' surface from the Freesurfer 'MNIAverage' directory.
    #     """
    #     if not op.exists(self._vol_src_path):
    #         logging.info("Generating volume source space...")

    #         self.vol_src = mne.setup_volume_source_space(
    #             subject="MNIAverage",
    #             subjects_dir=self.fs_subjects_dir,
    #             surface=surface,
    #             mindist=0,
    #             add_interpolator=False,  # Just for speed!
    #         )
    #         self.vol_src.save(self._vol_src_path, overwrite=True, verbose=True)
    #     else:
    #         logging.info("Reading volume source space from disk")
    #         self.vol_src = mne.read_source_spaces(self._vol_src_path, verbose=False)

    # Same for BEM
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
            self.bem = mne.make_bem_solution(model)
            mne.write_bem_solution(
                self._bem_solution_path, self.bem, overwrite=True, verbose=True
            )
        else:
            logging.info("Loading boundary element model from disk | %s", self.name)
            self.bem = mne.read_bem_solution(self._bem_solution_path, verbose=False)

    # Read manually aligned fiducials
    def _set_fiducials(self):
        """Internal method to read manually aligned fiducials."""
        self.fiducials, _coordinate_frame = mne.io.read_fiducials(self.fiducials_path)
        logging.debug("Fiducials coordinate frame: %s", _coordinate_frame)

    def _set_head_mri_t(self):
        """Internal method to read manually aligned fiducials."""
        self.head_mri_t = mne.read_trans(self.head_mri_t_path)

    def _set_info(self):
        """Internal method to read manually aligned fiducials."""
        self.info = mne.io.read_info(self.info_path)

    def _set_wm(self):
        """Internal method to read manually aligned fiducials."""
        self.wm = nib.load(self.wm_path)

    def _set_t1(self):
        """Internal method to read manually aligned fiducials."""
        self.t1 = nib.load(self.t1_path)

    def src_vertex_index_to_mri(
        self, idx: Optional[int or np.ndarray] = None
    ) -> np.ndarray:
        """
        Converts an index or set of indices from the source space
        to a point or set of points in MRI space

        Args:
            idx (int | np.ndarray): The index or indices in the source space to be converted.

        Returns:
            np.ndarray: A numpy array containing three float values
            representing the RAS coordinates of the given index.
        """
        assert (
            isinstance(idx, int) or isinstance(idx, np.ndarray) or isinstance(idx, list)
        ), f"vertex_index_to_mri argument idx should be of type int or np.ndarray, not type(idx)={type(idx)}"
        if isinstance(idx, int):
            return self.src["rr"][idx, :]
        else:
            return self.src["rr"][idx]

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

    def center_lia(self, pts: np.ndarray) -> np.ndarray:
        return mne.transforms.apply_trans(self.t1.affine, pts)

    def inverse_center_lia(self, pts: np.ndarray) -> np.ndarray:
        return mne.transforms.apply_trans(np.linalg.inv(self.t1.affine), pts)

    # def apply_mri_head_t(self, pts: np.ndarray) -> np.ndarray:
    #     return mne.transforms.apply_trans(self.head_mri_t, pts)

    def get_src_space_ras_nzo(
        self, transform: Optional[mne.Transform] = None
    ) -> np.ndarray:
        """
        Retrieves the source space coordinates in RAS with non-zero origin, with an optional transformation.

        Args:
            transform (Optional[mne.Transform]): An optional transformation to
            apply to the source space coordinates.
        Returns:
            np.ndarray: The source space coordinates in RAS (non-zero origin).
        """
        src_space = self.mri_to_ras_nzo(
            self.src["rr"]
        )  # Move volume from MRI space to RAS (non-zero origin)
        if transform is not None:
            src_space = mne.transforms.apply_trans(transform, src_space)

        # Whole src_space is a grid
        # Pick the actual points that conform the source space
        src_space = src_space[self.src["vertno"], :]

        return src_space

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
        # if transform is None:
        #     transform = self.head_mri_t
        # transform = mne.transforms.Transform("head", "mri", self.head_mri_t)
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
