#!/usr/bin/env python
"""
MNEBEM submodule for cerebra_atlas_python
"""
import logging
from functools import cached_property
import os.path as op
from typing import Tuple
import mne
import numpy as np
from ..data._cache import cache_pkl, cache_mne_bem

logger = logging.getLogger(__name__)


class BEMMNE:
    def __init__(self, cache_path: str, subjects_dir: str, **kwargs):
        self.bem_conductivity: Tuple[float, float, float] = (0.33, 0.0042, 0.33)
        self.bem_ico: int = 4
        self._bem_model: dict | None = None
        self._bem = None
        self.subjects_dir = subjects_dir

        self.bem_folder_path = op.join(self.subjects_dir, "bem")
        self.bem_conductivity_string = (
            "bem_" + "".join([str(x) + "_" for x in self.bem_conductivity])[:-1]
        )
        self.bem_string = f"{self.bem_conductivity_string}_ico_{self.bem_ico}"
        self._bem_path = op.join(cache_path, f"{self.bem_string}_bem.fif")
        # Output paths
        self._bem_model_path = op.join(
            cache_path,
            f"{self.bem_string}.pkl",
        )
        # Mapping for making sure bem indices refer to the same surface every time
        self.bem_layer_names = {1: "outer_skin", 2: "outer_skull", 3: "inner_skull"}

    # * PROPERTIES
    @cached_property
    def bem_model(self, subject_name="icbm152"):
        def compute_fn(self):
            logger.debug("Generating boundary element model... | %s", self.bem_string)
            return mne.make_bem_model(
                subject=subject_name,
                ico=self.bem_ico,
                conductivity=self.bem_conductivity,
                subjects_dir=self.subjects_dir,
            )

        return cache_pkl(compute_fn, self._bem_model_path, self)

    @cached_property
    def bem(self):
        def compute_fn(self):
            bem = mne.make_bem_solution(self.bem_model)
            return bem

        return cache_mne_bem(compute_fn, self._bem_path, self)

    # * METHODS
    def get_bem_vertices_mri(self) -> np.ndarray:
        return np.array([surf["rr"] for surf in self.bem_model])

    def get_bem_normals_mri(self) -> np.ndarray:
        return np.array([surf["nn"] for surf in self.bem_model])

    def get_bem_triangles(self) -> np.ndarray:
        return np.array([surf["tris"] for surf in self.bem_model])
