"""This module provides classes and functions for configuring and managing MNIAverage 
surfaces and related data processing operations."""
import os
import os.path as op
import logging
from abc import ABC, abstractmethod

import mne
import numpy as np

from cerebra_atlas_python.utils import read_config_as_dict


class BaseConfig(ABC):  # Abstract class
    """
    BaseConfig serves as an abstract base class for configuration management, designed to:
    - Read configuration settings from a file.
    - Use default values if the file is not available or lacks certain settings.
    - Allow customization through runtime parameters.
    """

    @abstractmethod
    def __init__(
        self, parent_name: str, default_config: (dict or None) = None, **kwargs
    ):
        """
        Initialize the BaseConfig object.

        Args:
            parent_name (str): Name of the parent class.
            default_config (dict or None, optional): Default configuration values.
                Used if the configuration file is missing or incomplete. Defaults to None.
            **kwargs: Additional keyword arguments to override both the configuration file
                and default values.
        """
        default_config = default_config or {}

        # Attempt to read the configuration
        config, config_success = read_config_as_dict(section=parent_name)

        # Choose config.ini over default_config
        if not config_success:
            # Use the provided default configuration if file reading is unsuccessful
            config = default_config
            if not config:
                logging.warning(
                    "Config and default values were not provided for class %s",
                    parent_name,
                )

        # Update missing keys in the config with default values
        for key, value in default_config.items():
            if key not in config:
                logging.warning(
                    "Value for variable %s not provided in %s. Defaulting to %s=%s",
                    key,
                    parent_name,
                    key,
                    value,
                )
            config.setdefault(key, value)

        # Override with any provided kwargs
        config.update(kwargs)

        # Set each configuration item as an instance attribute
        for key, value in config.items():
            try:
                # Attempt to evaluate the value if it's a string that represents a Python literal
                parsed_value = eval(value)
            except Exception:
                # Retain the original string value if eval fails
                parsed_value = value
            setattr(self, key, parsed_value)
        # Store the parent's name
        self._baseconfig_parent_name = parent_name

    @property
    def name(self) -> str:
        """Returns the name of the parent class.
        Returns:
            str: Name of the parent class
        """
        return self._baseconfig_parent_name


######### Computed using FreeSurfer "MNIAverage" surfaces generated previosly
class MNIAverage(BaseConfig):
    def __init__(self, **kwargs):
        default_config = {
            "mniaverage_output_path": "./generated/models",
            "fs_subjects_dir": os.getenv("SUBJECTS_DIR"),
            "bem_conductivity": (0.33, 0.0042, 0.33),
            "bem_ico": 4,
            "download_data": True,
        }
        super().__init__(
            parent_name=self.__class__.__name__, default_config=default_config
        )

        # If output folder does not exist, create it
        if not op.exists(self.mniaverage_output_path):
            os.makedirs(self.mniaverage_output_path, exist_ok=True)

        # Output paths
        self.vol_src_path = op.join(
            self.mniaverage_output_path, f"{self.__class__.__name__}-v-src.fif"
        )
        self.bem_path = op.join(
            self.mniaverage_output_path,
            f"{self.name}.fif",
        )

        self.vol_src = None
        self.bem = None
        self.fiducials = None

        self._set_bem()
        self._set_vol_src()
        self._set_fiducials()

    @property
    def bem_conductivity_string(self):
        return "bem_" + "".join([str(x) + "_" for x in self.bem_conductivity])[:-1]

    @property
    def name(self):
        return f"{super().name}_{self.bem_conductivity_string}_ico_{self.bem_ico}"

    @property
    def src(self):
        return self.vol_self.src[0]

    # If vol src fif does not exist, create it, otherwise read it
    def _set_vol_src(self):
        if not op.exists(self.vol_src_path):
            logging.info(f"Generating volume source space...")
            surface = op.join(
                self.fs_subjects_dir, "MNIAverage", "bem", "inner_skull.surf"
            )
            self.vol_src = mne.setup_volume_source_space(
                "MNIAverage",
                surface=surface,
                add_interpolator=False,  # Just for speed!
            )
            self.vol_src.save(self.vol_src_path, overwrite=True, verbose=True)
        else:
            logging.info(f"Reading volume source space from disk")
            self.vol_src = mne.read_source_spaces(self.vol_src_path)

    # Same for BEM
    def _set_bem(self):
        if not op.exists(self.bem_path):
            logging.info(f"Generating boundary element model... | {self.name}")
            model = mne.make_bem_model(
                subject="MNIAverage",
                ico=self.bem_ico,
                conductivity=self.bem_conductivity,
            )  # subjects_dir is env variable
            self.bem = mne.make_bem_solution(model)
            mne.write_bem_solution(
                self.bem_path, self.bem, overwrite=True, verbose=True
            )
        else:
            logging.info(f"Loading boundary element model from disk | {self.name}")
            self.bem = mne.read_bem_solution(self.bem_path)

    # Read manually aligned fiducials
    def _set_fiducials(self):
        self.fiducials, _coordinate_frame = mne.io.read_fiducials(
            f"{self.fs_subjects_dir}/MNIAverage/bem/MNIAverage-fiducials.fif"
        )

    # Single index from src space to RAS
    def index_to_ras(self, idx):
        return (
            mne.transforms.apply_trans(self.src["mri_ras_t"], self.src["rr"][idx, :])
            * 1000
        )

    def get_src_volume(self, transform=None):
        pts = mne.transforms.apply_trans(self.src["mri_ras_t"], self.src["rr"]) * 1000
        if transform is not None:
            pts = mne.transforms.apply_trans(transform, pts).astype(int)
        src_space = np.zeros((256, 256, 256)).astype(int)
        for i, pt in enumerate(pts):
            x, y, z = pt
            if i in self.src["vertno"]:
                src_space[x, y, z] = 1  # Usable source space
            else:
                src_space[x, y, z] = 2  # Box around source space
        return src_space

    def get_bem_surfaces(self, transform=None):
        surfaces = []
        for surf in self.bem["surfs"]:
            pts = mne.transforms.apply_trans(self.src["mri_ras_t"], surf["rr"]) * 1000
            if transform is not None:
                pts = mne.transforms.apply_trans(transform, pts).astype(int)
            surfaces.append(pts)
        surfaces = np.array(surfaces)
        return surfaces


if __name__ == "__main__":
    mniAverage = MNIAverage(bem_ico=4)
