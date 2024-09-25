#!/usr/bin/env python
"""
CerebraData
"""

import logging
import os.path as op
from functools import cached_property
from typing import Dict, Tuple
import numpy as np

from ._cache import cache_np, cache_pkl
from .labels import Labels
from .image import Image
from .freesurfer import FreeSurfer

logger = logging.getLogger(__name__)


class CerebraData(Labels, Image, FreeSurfer):
    """
    Super class that inherits from Labels, Image, and FreeSurfer classes

    Attributes:
        cerebra_data_path (str): Path to cerebra data dir
        cache_path_cerebra (str): Path to cerebra cache dir
    """

    def __init__(self, cache_path: str, data_path=None, **kwargs):
        """Instantiates sub classes"""

        self.cerebra_data_path = (
            op.dirname(__file__) + "/cerebra_data" if data_path is None else data_path
        )

        Labels.__init__(self, cerebra_data_path=self.cerebra_data_path, **kwargs)
        Image.__init__(self, cerebra_data_path=self.cerebra_data_path, **kwargs)
        FreeSurfer.__init__(self, cerebra_data_path=self.cerebra_data_path, **kwargs)

        self.cache_path_cerebra: str = cache_path
        # Output paths
        self._cerebra_volume_path = op.join(
            self.cache_path_cerebra, "cerebra_volume.npy"
        )
        self._cerebra_volume_lia_path = op.join(
            self.cache_path_cerebra, "cerebra_volume.npy"
        )
        self._cerebra_sparse_path = op.join(
            self.cache_path_cerebra, "CerebrA_sparse.pkl"
        )

    @cached_property
    def cerebra_volume(self) -> np.ndarray:
        """(256,256,256) voxel volume containg cerebra data in RAS space"""
        return cache_np(self._get_wm_cerebra_volume_ras, self._cerebra_volume_path)

    @cached_property
    def cerebra_volume_lia(self) -> np.ndarray:
        """(256,256,256) voxel volume containg cerebra data in LIA space"""
        return cache_np(self._get_wm_cerebra_volume_lia, self._cerebra_volume_lia_path)

    @cached_property
    def affine(self) -> np.ndarray:
        """(4,4) affine matrix"""
        _, _affine = self.get_cerebra_vox_affine_ras()
        return _affine

    @cached_property
    def cerebra_sparse(self) -> Dict[int, np.ndarray]:
        """Dictionary containing sparse voxel grid for each region in RAS space[1-103]"""

        def compute_fn(self):
            return {
                region_id: self._calculate_points_from_region_id(region_id)
                for region_id in self.region_ids
            }

        return cache_pkl(compute_fn, self._cerebra_sparse_path, self)

    def _calculate_points_from_region_id(self, region_id):
        # Find points for each region. Should only be called once per region
        # Then, a sparse representation of the region data is stored/loaded as a .npy file
        return np.array(np.where(self.cerebra_volume == region_id), dtype=np.uint8).T

    def _get_wm_filled_cerebra_volume_aff_ras(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get (256,256,256) volume filled with wm. Used internally. To access the volume (RAS) use cerebra_volume attribute"""
        cerebra_vox_ras, affine = self.get_cerebra_vox_affine_ras()
        wm_vox_ras, _ = self.get_wm_vox_affine_ras()
        cerebra_vox_ras[(wm_vox_ras != 0) & (cerebra_vox_ras == 0)] = 103
        return cerebra_vox_ras.astype(int), affine

    def _get_wm_filled_cerebra_volume_aff_lia(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get (256,256,256) volume filled with wm. Used internally. To access the volume (LIA) use cerebra_volume attribute"""
        cerebra_vox_lia, affine = self.get_cerebra_vox_affine_lia()
        wm_vox_lia, _ = self.get_wm_vox_affine_lia()
        cerebra_vox_lia[(wm_vox_lia != 0) & (cerebra_vox_lia == 0)] = 103
        return cerebra_vox_lia.astype(int), affine

    def _get_wm_cerebra_volume_ras(self) -> np.ndarray:
        """Get cerebra volume filled with wm in RAS space"""
        return self._get_wm_filled_cerebra_volume_aff_ras()[0]

    def _get_wm_affine_ras(self) -> np.ndarray:
        """Get cerebra affine in RAS space"""
        return self._get_wm_filled_cerebra_volume_aff_ras()[1]

    def _get_wm_cerebra_volume_lia(self) -> np.ndarray:
        """Get cerebra volume filled with wm in RAS space"""
        return self._get_wm_filled_cerebra_volume_aff_lia()[0]

    def _get_wm_affine_lia(self) -> np.ndarray:
        """Get cerebra affine in RAS space"""
        return self._get_wm_filled_cerebra_volume_aff_lia()[1]

    def get_region_centroid(self, region_id):
        centroid = np.round(self.cerebra_sparse[region_id].mean(axis=0)).astype(
            np.uint8
        )

        return centroid
