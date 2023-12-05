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
import numpy as np

from cerebra_atlas_python.config import BaseConfig


class MNIAverage(BaseConfig):
    """
    A class for generating and managing MNIAverage Freesurfer surfaces in MNE-Python applications.

    MNIAverage handles the creation and management of MNIAverage surfaces, including volume source spaces,
    boundary element models (BEM), and fiducials. It provides functionality for converting indices to RAS coordinates
    and generating source volumes and BEM surfaces.

    Attributes:
        mniaverage_output_path (str): Path to the output directory for MNIAverage data.
        fs_subjects_dir (str): Directory path where Freesurfer subjects are stored.
        bem_conductivity (tuple[float, float, float]): Conductivity values for the BEM model.
        bem_ico (int): The icosahedron level for the BEM model.
        download_data (bool): Flag to indicate whether data should be downloaded.
        vol_src (mne.source_space.SourceSpaces): Volume source space object.
        bem (mne.bem.ConductorModel): Boundary element model object.
        fiducials (List[mne.io._digitization.DigPoint]): Fiducial points object.

    Methods:
        __init__: Constructor to initialize the MNIAverage class.
        name: Property to get the class name with additional configuration details.
        src: Property to access the source space.
        _set_vol_src: Internal method to set up the volume source space.
        _set_bem: Internal method to set up the boundary element model.
        _set_fiducials: Internal method to set up fiducials.
        index_to_ras: Convert an index in the source space to RAS coordinates.
        get_src_volume: Get the source volume in a numpy array.
        get_bem_surfaces: Get the surfaces of the boundary element model.
    """

    def __init__(self, config_path=op.dirname(__file__) + "/config.ini", **kwargs):
        self.mniaverage_output_path: str = None
        self.fs_subjects_dir: str = None
        self.bem_conductivity: Tuple[float, float, float] = None
        self.bem_ico: int = None
        self.default_data_path: str = None
        default_config = {
            "mniaverage_output_path": "./generated/models",
            "fs_subjects_dir": os.getenv("SUBJECTS_DIR"),
            "bem_conductivity": (0.33, 0.0042, 0.33),
            "bem_ico": 4,
            "default_data_path": op.dirname(__file__) + "/cerebra_data/MNIAverage",
        }

        super().__init__(
            config_path=config_path,
            parent_name=self.__class__.__name__,
            default_config=default_config,
            **kwargs,
        )
        self.vol_src: Optional[mne.SourceSpaces] = None
        self.bem: Optional[mne.bem.ConductorModel] = None
        self.fiducials: Optional[List[mne.io._digitization.DigPoint]] = None

        # If output folder does not exist, create it
        if not op.exists(self.mniaverage_output_path):
            os.makedirs(self.mniaverage_output_path, exist_ok=True)

        # self.default_data_path =

        # TODO: Data priority: Default data path -> Then freesurfer dir -> Then error
        if not self.fs_subjects_dir:
            logging.info(
                "Freesurfer subjects folder not found, using default data path"
            )
            self.fs_subjects_dir = self.default_data_path
        elif "MNIAverage" not in os.listdir(self.fs_subjects_dir):
            logging.info("MNIAverage subject data not found, using default data path")
            self.fs_subjects_dir = self.default_data_path

        else:
            logging.info("Using data from SUBJECTS_DIR/MNIAverage")

        # Output paths
        self._vol_src_path = op.join(
            self.mniaverage_output_path,
            f"mne.SourceSpaces{self.__class__.__name__}-v-src.fif",
        )
        self._bem_path = op.join(
            self.mniaverage_output_path,
            f"{self.name}.fif",
        )

        # Mapping for making sure bem indices refer to the same surface every time
        self.bem_names = {1: "outer_skin", 2: "outer_skull", 3: "inner_skull"}

        self._set_bem()
        self._set_vol_src()
        self._set_fiducials()

    @property
    def bem_conductivity_string(self) -> str:
        """
        Property to get the BEM conductivity values as a formatted string.

        This property formats the BEM conductivity values as a string, which is useful
        for creating filenames or labels that include conductivity information.

        Returns:
            str: A string representing the BEM conductivity values, formatted as 'bem_value1_value2_value3'.
        """
        return "bem_" + "".join([str(x) + "_" for x in self.bem_conductivity])[:-1]

    @property
    def name(self) -> str:
        """
        Property to get the enhanced name of the class instance.

        This property extends the base name from the parent class with additional details
        about BEM conductivity and icosahedron level, making it specific to the current configuration.

        Returns:
            str: Enhanced name of the class instance, including BEM conductivity and icosahedron level.
        """
        return f"{super().name}_{self.bem_conductivity_string}_ico_{self.bem_ico}"

    @property
    def src(self) -> Dict:
        """
        This property provides convenient access to the first source space object, assuming
        the volume source space has been set.

        Returns:
            The first source space object, if available.
        """
        return self.vol_src[0]

    # If vol src fif does not exist, create it, otherwise read it
    def _set_vol_src(self):
        """
        Internal method to set up the volume source space.

        This method checks if the volume source space file exists. If not, it generates a new
        volume source space and saves it to the specified path. If the file exists, it reads the
        source space from the disk.

        Uses the 'inner_skull.surf' surface from the Freesurfer 'MNIAverage' directory.
        """
        if not op.exists(self._vol_src_path):
            logging.info("Generating volume source space...")
            surface = op.join(
                self.fs_subjects_dir, "MNIAverage", "bem", "inner_skull.surf"
            )
            self.vol_src = mne.setup_volume_source_space(
                subject="MNIAverage",
                subjects_dir=self.fs_subjects_dir,
                surface=surface,
                mindist=0,
                add_interpolator=False,  # Just for speed!
            )
            self.vol_src.save(self._vol_src_path, overwrite=True, verbose=True)
        else:
            logging.info("Reading volume source space from disk")
            self.vol_src = mne.read_source_spaces(self._vol_src_path, verbose=False)

    # Same for BEM
    def _set_bem(self):
        """
        Internal method to set up the boundary element model (BEM).

        This method checks if the BEM model file exists. If not, it generates a new BEM model
        using the specified conductivity and icosahedron level, and saves it. If the file exists,
        it reads the BEM model from the disk.

        The BEM model is associated with the 'MNIAverage' subject in Freesurfer.
        """
        if not op.exists(self._bem_path):
            logging.info("Generating boundary element model... | %s", self.name)
            model = mne.make_bem_model(
                subject="MNIAverage",
                ico=self.bem_ico,
                conductivity=self.bem_conductivity,
                subjects_dir=self.fs_subjects_dir,
            )
            self.bem = mne.make_bem_solution(model)
            mne.write_bem_solution(
                self._bem_path, self.bem, overwrite=True, verbose=True
            )
        else:
            logging.info("Loading boundary element model from disk | %s", self.name)
            self.bem = mne.read_bem_solution(self._bem_path, verbose=False)

    # Read manually aligned fiducials
    def _set_fiducials(self):
        """
        Internal method to read manually aligned fiducials.

        Reads the fiducial points from a specified file in the Freesurfer 'MNIAverage' directory,
        used for aligning the head model with the MEG/EEG sensors.
        """
        self.fiducials, _coordinate_frame = mne.io.read_fiducials(
            f"{self.fs_subjects_dir}/MNIAverage/bem/MNIAverage-fiducials.fif"
        )

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
        return mne.transforms.apply_trans(self.src["mri_ras_t"], pts) * 1000

    def get_src_space_ras_nzo(
        self, transform: Optional[mne.transforms.Transform] = None
    ) -> np.ndarray:
        """
        Retrieves the source space coordinates in RAS with non-zero origin, with an optional transformation.

        Args:
            transform (Optional[mne.transforms.Transform]): An optional transformation to
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
        self, transform: Optional[mne.transforms.Transform] = None
    ) -> np.ndarray:
        """
        Retrieves the BEM surfaces in RAS coordinates with non-zero origin, with an optional transformation.

        Args:
            transform (Optional[mne.transforms.Transform]): An optional transformation to apply to the BEM surfaces.

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
        surfaces = np.array(surfaces)
        return surfaces


if __name__ == "__main__":
    mniAverage = MNIAverage(bem_ico=4)
