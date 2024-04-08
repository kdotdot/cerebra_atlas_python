import logging
import mne
import os.path as op
import numpy as np
from .config import Config
from .cerebra import CerebrA
from .mni_average import MNIAverage
from .utils import (
    move_volume_from_lia_to_ras,
    find_closest_point,
    merge_voxel_grids,
    point_cloud_to_voxel,
    move_volume_from_ras_to_lia,
    get_volume_ras,
)
from .cache import cache_mne_src

logger = logging.getLogger(__name__)

# Downsampled & sparse version of the cerebra volume
class CerebraMNE(CerebrA, MNIAverage, Config):
    def __init__(self,config_path=None,**kwargs,):
        Config.__init__(self, class_name=self.__class__.__name__,config_path=config_path, **kwargs)
        MNIAverage.__init__(self,config_path=config_path,**kwargs,)
        CerebrA.__init__(self,config_path=config_path,**kwargs,)

        self._bem_surfaces: np.ndarray = None
        self._bem_volume: np.ndarray = None


        # Load on demand [using @property] (slow)
        self._src_space_path = op.join(
            self.cerebra_output_path, f"{self.src_space_string}_src.fif"
        )
        self._src_space: mne.SourceSpaces = None


    # * PROPERTIES
    @property
    @cache_mne_src()
    def src_space(self):
        def compute_fn(self):
            logger.info("Generating new source space %s {self._src_space_path}")
            normals = np.repeat([[0, 0, 1]], self.src_space_n_total_points, axis=0)

            rr = point_cloud_to_voxel(self.src_space_points)
            rr = move_volume_from_ras_to_lia(rr)
            rr = np.argwhere(rr != 0)  # Back to point cloud
            inv_aff = np.linalg.inv(self.t1.affine)
            # Translation
            inv_aff[:, 3][2] = 132
            # Rotation
            inv_aff[:, 1][2] *= -1
            inv_aff[:, 2][1] *= -1
            inv_aff[:, 3][1] = -128
            rr = mne.transforms.apply_trans(inv_aff, rr)
            rr = rr / 1000
            pos = dict(rr=rr, nn=normals)
            src_space = mne.setup_volume_source_space(pos=pos)
            return src_space
        return compute_fn, self._src_space_path

    @property
    def bem_surfaces(self):
        if self._bem_surfaces is None:
            self._set_bem_surfaces()
        return self._bem_surfaces

    @property
    def bem_volume(self):
        if self._bem_volume is None:
            self._set_bem_volume()
        return self._bem_volume
    
    def _set_bem_volume(self):
        self._bem_volume = None
        for surf, bem_id in zip(self.bem_surfaces, self.bem_names.keys()):
            if self._bem_volume is None:
                self._bem_volume = point_cloud_to_voxel(surf, vox_value=bem_id)
            else:
                self._bem_volume = merge_voxel_grids(
                    self.bem_volume, point_cloud_to_voxel(surf, vox_value=bem_id)
                )

    def _set_bem_surfaces(self):
        self._bem_surfaces = self.get_bem_surfaces_ras_nzo(
            transform=self.affine
        )

    def prepare_plot_data_2d(self, plot_t1_volume=False, **kwargs):
        data = super().prepare_plot_data_2d(**kwargs)
        if plot_t1_volume:
            data["t1_volume"] = move_volume_from_lia_to_ras(self.t1.dataobj)
        return data

    def plot_data_2d(self, plot_t1_volume=False, **kwargs):
        t1_volume = None
        if plot_t1_volume:
            t1_volume = move_volume_from_lia_to_ras(self.t1.dataobj)
        return super().plot_data_2d(t1_volume=t1_volume,**kwargs)