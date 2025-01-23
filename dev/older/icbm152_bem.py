"""
Handles Freesurfer related operations
"""

from typing import Tuple
import os.path as op
import logging
import numpy as np
import mne
import appdirs
from .config import Config
from .__cache import cache_pkl, cache_mne_bem

logger = logging.getLogger(__name__)


class ICBM152BEM(Config):
    """
    Reads data from FreeSurfer/subjects/icbm152
    """

    def __init__(self, **kwargs):
        self.cache_path_icbm: str = op.join(
            appdirs.user_cache_dir("cerebra_atlas_python"), "ICBM152"
        )
        self.cerebra_data_path: str = op.dirname(__file__) + "/cerebra_data"
        self.bem_conductivity: Tuple[float, float, float] = (0.33, 0.0042, 0.33)
        self.bem_ico: int = 4
        Config.__init__(
            self,
            class_name=self.__class__.__name__,
            **kwargs,
        )
        self._bem_model: dict = None
        self._bem = None

        self.bem_folder_path = op.join(self.subject_dir, "bem")
        self.bem_conductivity_string = (
            "bem_" + "".join([str(x) + "_" for x in self.bem_conductivity])[:-1]
        )
        self.bem_name = f"{self.bem_conductivity_string}_ico_{self.bem_ico}"
        self._bem_path = op.join(self.cache_path_icbm, f"{self.bem_name}_bem.fif")
        # Output paths
        self._bem_model_path = op.join(
            self.cache_path_icbm,
            f"{self.bem_name}.pkl",
        )
        # Mapping for making sure bem indices refer to the same surface every time
        self.bem_layer_names = {1: "outer_skin", 2: "outer_skull", 3: "inner_skull"}

    # * PROPERTIES
    @property
    @cache_pkl()
    def bem_model(self):
        def compute_fn(self):
            logging.debug("Generating boundary element model... | %s", self.bem_name)
            return mne.make_bem_model(
                subject=self.subject_name,
                ico=self.bem_ico,
                conductivity=self.bem_conductivity,
                subjects_dir=self.subjects_dir,
            )

        return compute_fn, self._bem_model_path

    @property
    @cache_mne_bem()
    def bem(self):
        def compute_fn(self):
            bem = mne.make_bem_solution(self.bem_model)
            return bem

        return compute_fn, self._bem_path

    # * METHODS
    def get_bem_vertices_mri(self) -> np.ndarray:
        return np.array([surf["rr"] for surf in self.bem_model])

    def get_bem_normals_mri(self) -> np.ndarray:
        return np.array([surf["nn"] for surf in self.bem_model])

    def get_bem_triangles(self) -> np.ndarray:
        return np.array([surf["tris"] for surf in self.bem_model])
