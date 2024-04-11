import logging
from typing import Tuple, Optional, List
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
from .cache import cache_mne_src, cache_mne_bem

logger = logging.getLogger(__name__)

# Uses MNE
class CerebraMNE(CerebrA, MNIAverage, Config):
    def __init__(self,config_path=None,**kwargs,):
        Config.__init__(self, class_name=self.__class__.__name__,config_path=config_path, **kwargs)
        MNIAverage.__init__(self,config_path=config_path,**kwargs,)
        CerebrA.__init__(self,config_path=config_path,**kwargs,)

        self._bem_surfaces: np.ndarray = None
        self._bem_volume: np.ndarray = None

        # Always load (fast/required)
        self.fiducials: Optional[List[mne.io._digitization.DigPoint]] = None
        self.head_mri_t: Optional[mne.Transform] = None
        self.mri_ras_t: Optional[mne.Transform] = None

        # Load on demand [using @property] (slow)
        self._src_space: mne.SourceSpaces = None
        self._bem: mne.bem.ConductorModel = None

 
        self._src_space_path = op.join(
            self.cerebra_output_path, f"{self.src_space_string}_src.fif"
        )
        self._bem_path = op.join(
            self.cerebra_output_path, f"{self.bem_name}_bem.fif"
        )

        self._set_fiducials()
        self._set_head_mri_t()
        self._set_mri_ras_t()

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
    @cache_mne_bem()
    def bem(self):
        def compute_fn(self):
            bem = mne.make_bem_solution(self.bem_model)
            return bem
        return compute_fn, self._bem_path


    @property
    def bem_surfaces(self):
        if self._bem_surfaces is None:
            self._bem_surfaces
        return self._bem_surfaces

    @property
    def bem_volume(self):
        if self._bem_volume is None:
            self._set_bem_volume()
        return self._bem_volume
    
    # * SETTERS
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
        res = mne.transforms.apply_trans(self.mri_ras_t["trans"], pts) * 1000
        return res

    def ras_nzo_to_mri(self, pts: np.ndarray) -> np.ndarray:
        return (
            mne.transforms.apply_trans(np.linalg.inv(self.mri_ras_t["trans"]), pts)
            / 1000
        )
    
    def apply_affine(self, pts: np.ndarray) -> np.ndarray:
        return mne.transforms.apply_trans(self.affine, pts).astype(int)
    
    def apply_inverse_affine(self, pts: np.ndarray) -> np.ndarray:
        return mne.transforms.apply_trans(np.linalg.inv(self.affine), pts)
    
    def mri_to_ras(self, pts: np.ndarray) -> np.ndarray:
        pts = self.mri_to_ras_nzo(pts)
        return self.apply_affine(pts)
    
    def ras_to_mri(self, pts: np.ndarray) -> np.ndarray:
        pts = self.apply_inverse_affine(pts)
        return self.ras_nzo_to_mri(pts)
    
    def apply_head_mri_t(self, pts: np.ndarray) -> np.ndarray:
        return mne.transforms.apply_trans(self.head_mri_t, pts) / 1000
    
    # * METHODS

    def get_fiducials_mri(self) -> np.ndarray:
        return np.array([f["r"] for f in self.fiducials])
    
    def get_fiducials_ras(self) -> np.ndarray:
        return self.mri_to_ras(self.get_fiducials_mri())
    
    def get_bem_vertices_mri(self) -> np.ndarray:
        return np.array([surf["rr"] for surf in self.bem_model])
    
    def get_bem_vertices_ras(self) -> np.ndarray:
        return [self.mri_to_ras(surf) for surf in self.get_bem_vertices_mri()]

    def get_bem_normals_mri(self) -> np.ndarray:
        return np.array([surf["nn"] for surf in self.bem_model])
    
    def get_bem_normals_ras(self) -> np.ndarray:
        return [self.mri_to_ras(normals) for normals in self.get_bem_normals_mri()]
    
    def get_bem_triangles(self) -> np.ndarray:
        return np.array([surf["tris"] for surf in self.bem_model])
        


    def prepare_plot_data_2d(self, plot_t1_volume=False, **kwargs):
        data = super().prepare_plot_data_2d(**kwargs)
        if plot_t1_volume:
            data["t1_volume"] = self.get_t1_volume()
        return data

    def plot_data_2d(self, plot_t1_volume=False, **kwargs):
        t1_volume = None
        if plot_t1_volume:
            t1_volume = self.get_t1_volume()
        return super().plot_data_2d(t1_volume=t1_volume,**kwargs)