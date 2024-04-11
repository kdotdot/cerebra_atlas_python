"""
MNIAverage
"""
import os
import os.path as op
import logging
from typing import Tuple, Optional, List

import nibabel as nib
import numpy as np

from .config import Config
from .cache import cache_pkl
from .utils import (
    move_volume_from_lia_to_ras,
    find_closest_point,
    merge_voxel_grids,
    point_cloud_to_voxel,
    move_volume_from_ras_to_lia,
    get_volume_ras,
)

# Uses freesurfer
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

        # Load on demand [using @property] (slow/rarely used)
        #self._bem: Optional[mne.bem.ConductorModel] = None
        self._bem_model: dict = None
        self._t1 = None
        self._wm = None
        self._info = None

        # If output folder does not exist, create it
        if not op.exists(self.mniaverage_output_path):
            os.makedirs(self.mniaverage_output_path, exist_ok=True)

        # Input paths
        self.subject_name = "icbm152"
        self.subjects_dir = op.join(self.cerebra_data_path, "FreeSurfer/subjects")
        self.subject_dir = op.join(self.subjects_dir, self.subject_name)
        self.bem_folder_path = op.join(self.subject_dir, "bem")
        self.fiducials_path = op.join(
            self.cerebra_data_path, f"{self.subject_name}-fiducials.fif"
        )
        self.wm_path = op.join(self.subject_dir, "mri/wm.mgz")
        self.t1_path = op.join(self.subject_dir, "mri/T1.mgz")
        self.head_mri_t_path = op.join(self.cerebra_data_path, "head_mri_t.fif")
        # self.info_path = op.join(self.cerebra_data_path, "info.fif")
        self.mri_ras_t_path = op.join(self.cerebra_data_path, "mri_ras-trans.fif")

        # Output paths
        self._bem_model_path = op.join(
            self.mniaverage_output_path,
            f"{self.bem_name}.pkl",
        )

        # Mapping for making sure bem indices refer to the same surface every time
        self.bem_layer_names = {1: "outer_skin", 2: "outer_skull", 3: "inner_skull"}

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
    def bem_name(self) -> str:
        """Property to get the enhanced name of the class instance.
        Returns:
            str: Enhanced name of the class instance, including BEM conductivity and icosahedron level.
        """
        return f"{self.bem_conductivity_string}_ico_{self.bem_ico}"

    @property
    @cache_pkl()
    def bem_model(self):
        def compute_fn(self):
            import mne
            logging.info("Generating boundary element model... | %s", self.bem_name)
            bem_model = mne.make_bem_model(
                subject=self.subject_name,
                ico=self.bem_ico,
                conductivity=self.bem_conductivity,
                subjects_dir=self.subjects_dir,
            )
            return bem_model
            
        return compute_fn, self._bem_model_path

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

    # @property
    # def info(self):
    #     if self._info is None:
    #         self._info = mne.io.read_info(self.info_path)
    #     return self._info

    def get_t1_volume_lia(self):
        return np.array(self.t1.dataobj)

    def get_t1_volume_ras(self):
        return move_volume_from_lia_to_ras(self.get_t1_volume_lia())
    



if __name__ == "__main__":
    mniAverage = MNIAverage(bem_ico=4)
