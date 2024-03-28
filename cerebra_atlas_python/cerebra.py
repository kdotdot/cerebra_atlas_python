"""
Cerebra...
"""
import os.path as op
import logging
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib

from .plotting import (
    plot_volume_3d,
    orthoview,
    plot_brain_slice_2d
)
from .utils import (
    find_closest_point  
    
)
from .config import Config
from .source_space import get_source_space_mask
from .cache import cache_pkl,cache_np 

logger = logging.getLogger(__name__)

class CerebrA(Config):
    def __init__(self, config_path=None, **kwargs,):
        self.cerebra_output_path: str = "./generated/cerebra"
        self.cerebra_data_path: str = op.dirname(__file__) + "/cerebra_data"

        self.source_space_grid_size: int = 3
        self.source_space_include_wm: bool = False
        self.source_space_include_non_cortical = True

        Config.__init__(self, class_name=self.__class__.__name__,config_path=config_path, **kwargs)

        # Always load (fast/required)
        self.label_details: pd.DataFrame = None
        self.cerebra_volume: np.ndarray = None
        self.affine: np.ndarray = None
        # Load on demand [using @property] (slow)
        self._cerebra_sparse: Dict[np.ndarray] = None
        self._src_space_points: np.ndarray = None
        self._src_space_mask: np.ndarray = None
        self._src_space_labels: np.ndarray = None
        self._src_space_n_points_per_region: np.ndarray = None


        # Input paths
        cerebra_volume_path = op.join(self.cerebra_data_path, "volume.npy")
        cerebra_affine_path = op.join(self.cerebra_data_path, "affine.npy")
        label_details_path = op.join(self.cerebra_data_path, "label_details.csv")


        # Output paths
        self._cerebra_sparse_path = op.join(self.cerebra_output_path, "CerebrA_sparse.pkl")
        self.src_space_string = f"src_space_{self.source_space_grid_size}mm{'wm' if self.source_space_include_wm else ''}{'_nc' if self.source_space_include_non_cortical else ''}"
        self._src_space_mask_path = op.join(
            self.cerebra_output_path, f"{self.src_space_string}_mask.npy"
        )
        self._src_space_points_path = op.join(
            self.cerebra_output_path, f"{self.src_space_string}_src_pts.npy"
        )

        self.cerebra_volume = np.load(cerebra_volume_path)
        self.affine = np.load(cerebra_affine_path)
        self.label_details = pd.read_csv(label_details_path, index_col=0)

        # Metadata
        self.region_ids = np.sort(self.label_details["CerebrA ID"].unique())

    @property
    @cache_pkl()
    def cerebra_sparse(self):
        def compute_fn(self):
            return {region_id: self.calculate_points_from_region_id(region_id) for region_id in self.region_ids}
        return compute_fn, self._cerebra_sparse_path

    @property
    @cache_np()
    def src_space_mask(self):
        def compute_fn(self):
            return get_source_space_mask(self, self.source_space_grid_size, self.source_space_include_wm, self.source_space_include_non_cortical)
        return compute_fn, self._src_space_mask_path

    @property
    @cache_np()
    def src_space_points(self):
        def compute_fn(self):
            return np.indices([256, 256, 256])[:, self.src_space_mask].T
        return compute_fn, self._src_space_points_path
    
    
    @property
    def src_space_labels(self):
        if self._src_space_labels is None:
            self._src_space_labels = self.cerebra_volume[self.src_space_mask]
        return self._src_space_labels

    @property
    def src_space_n_points_per_region(self):
        if self._src_space_n_points_per_region is None:
            points_per_region = np.zeros(len(self.region_ids))
            region_ids, counts = np.unique(self.src_space_labels, return_counts=True)
            points_per_region[region_ids] = counts
            self._src_space_n_points_per_region = points_per_region
        return self._src_space_n_points_per_region

    @property
    def src_space_n_total_points(self):
        return len(self.src_space_points)

    ######
    # * METHODS
   
    def get_region_id_from_point(self, point):
        point = np.array(point).astype(int)
        region_id = self.cerebra_volume[point[0], point[1], point[2]]
        return int(region_id)  # Can be get_points_from_region_id0 and 103

    def get_region_data_from_region_id(self, region_id):
        region_data = self.label_details[self.label_details["CerebrA ID"] == region_id]
        if len(region_data) < 1:
            return None
        return region_data

    def get_region_data_from_region_name(self, region_name):
        region_data = self.label_details[
            self.label_details["Label Name"] == region_name
        ]
        return region_data

    def get_region_id_from_region_name(self, region_name):
        region_data = self.label_details[
            self.label_details["Label Name"] == region_name
        ]
        return region_data["CerebrA ID"].item()

    def get_region_name_from_region_id(self, region_id):
        if region_id == 0:
            return "Empty region"
        if region_id > 103:
            return "Non-valid id"
        return self.label_details[self.label_details["CerebrA ID"] == region_id][
            "Label Name"
        ].item()

    # Given a point in a (256,256,256) 3d space (RAS), determines
    # which brain region it belongs to
    def get_region_name_from_point(self, point):
        region_id = self.get_region_id_from_point(point)
        region_name = self.get_region_name_from_region_id(region_id)
        return region_name

    # Find points for each region. Should only be called once per region
    # Then, a sparse representation of the region data is stored/loaded as a .npy file
    def calculate_points_from_region_id(self, region_id):
        return np.array(np.where(self.cerebra_volume == region_id), dtype=np.uint8).T

    def get_points_from_region_id(self, region_id):
        return self.cerebra_sparse[region_id]

    def get_n_points_from_region_id(self, region_id):
        if region_id == 0:
            return (self.cerebra_volume == 0).sum()
        else:
            return len(self.get_points_from_region_id(region_id))

    # Given a brain region name, get an array of points that
    # make up said region
    def get_points_from_region_name(self, region_name):
        region_id = self.get_region_id_from_region_name(region_name)
        return self.get_points_from_region_id(region_id)

    def get_distance_to_inner_skull(self, pt):
        raise NotImplementedError()
        # # TODO: see mne.bem.distance_to_bem
        # _, _, inner_skull_points = self.bem_surfaces
        # closest_point, distance = find_closest_point(inner_skull_points, pt)

        # return closest_point, distance

    def get_closest_region_to_whitematter(self, pt, n_max=20):
        success = False

        region_id = self.get_region_id_from_point(pt)
        if region_id != 103 and region_id != 0:
            logger.warning(
                "Attempting to get closest region to whitematter from a non-whitematter region -> %s",
                region_id,
            )
            return success, None, None

        # Search around the point
        for i in range(n_max):
            for inc_id in range(1, 8):
                inc = (
                    np.array(list(bin(inc_id).split("b")[-1].rjust(3, "0"))).astype(int)
                    * i
                )  # 001, 010 ... 111

                pt1 = pt.copy() + inc
                pt2 = pt.copy() - inc

                region1 = self.get_region_id_from_point(pt1)
                if region1 != 0 and region1 != 103:
                    success = True
                    return success, pt1, region1

                region2 = self.get_region_id_from_point(pt2)
                if region2 != 0 and region2 != 103:
                    success = True
                    return success, pt2, region2

        logger.warning(
            "Get_closest_region_to_whitematter unable to find close region %s", n_max
        )
        return success, pt, region_id
    
    def get_cortical_region_ids(self, hemisphere=None):

        mask = self.label_details["cortical"].fillna(False)
        if hemisphere is not None:
            if hemisphere == "left":
                mask = mask & (self.label_details["hemisphere"]=="Left")
            elif hemisphere == "right":
                mask = mask & (self.label_details["hemisphere"]=="Right")
            else:
                raise ValueError(f"Unknown hemisphere {hemisphere=}")
        
        cortical_labels = self.label_details[mask]["CerebrA ID"].to_numpy()
        return cortical_labels
    
    def get_non_cortical_region_ids(self, hemisphere=None):

        mask = ~(self.label_details["cortical"].fillna(True))
        non_cortical_labels = self.label_details[mask]["CerebrA ID"].to_numpy()
        return non_cortical_labels
    
    def get_visual_cortex_region_ids(self):
        
        perception_visual_ambient = np.array([9,31,60,82])

        return perception_visual_ambient

    def find_region_centroid_from_name(self, region_name):
        region_id = self.get_region_id_from_region_name(region_name)
        return self.find_region_centroid_from_id(region_id)

    def find_region_centroid_from_id(self, region_id):
        centroid = np.round(
            self.get_points_from_region_id(region_id).mean(axis=0)
        ).astype(np.uint8)

        if self.get_region_id_from_point(centroid) == 103:
            success, pt, closest_region_id = self.get_closest_region_to_whitematter(
                centroid
            )
            if not success:
                logger.error(
                    "Unable to find closest region to whitematter from region centroid"
                )
            elif closest_region_id != region_id:
                logger.warning("Region centroid is outside of region?")
            centroid = pt
        return centroid

    # * PLOTTING
    def prepare_plot_data_2d(
        self,
        plot_src_space: bool = False,
        plot_distance_to_inner_skull: bool = False,
        plot_highlighted_region: int = None,
        plot_cortical: bool = None,
        **kwargs,
    ):
        src_space_points = None
        if plot_src_space:
            src_space_points = self.src_space_points

        region_centroid = None
        if plot_highlighted_region is not None:
            assert isinstance(plot_highlighted_region, int)
            # region_pts = self.get_points_from_region_id(plot_highlighted_region)
            # region_volume = point_cloud_to_voxel(region_pts)
            reg_name = self.get_region_name_from_region_id(plot_highlighted_region)
            region_centroid = self.find_region_centroid_from_id(plot_highlighted_region)
            if "pt" not in kwargs:
                kwargs["pt"] = region_centroid
                pt_text = f"{region_centroid}\n{reg_name}"

        pt_dist = None
        if plot_distance_to_inner_skull:
            if "pt" not in kwargs:
                raise ValueError(
                    "pt=np.ndarray should be set if plot_distance_to_inner_skull=True"
                )
            pt = kwargs["pt"]
            pt_dist = self.get_distance_to_inner_skull(pt)

        pt_text = None
        if "pt" in kwargs and kwargs["pt"] is not None:
            reg_name = self.get_region_name_from_point(kwargs["pt"])
            pt_text = f"{kwargs['pt']}\n{reg_name}"

        volume_colors = None
        if plot_cortical:
            cortical_mask = self.label_details["cortical"]
            from .plotting import cortical_color, non_cortical_color
            volume_colors = [
                matplotlib.colors.to_rgb(cortical_color)
                if c
                else matplotlib.colors.to_rgb(non_cortical_color)
                for c in cortical_mask
            ]
            volume_colors = np.array(volume_colors)

        return (
            src_space_points,
            region_centroid,
            pt_dist,
            pt_text,
            volume_colors,
        )

    def plot_data_2d(
        self,
        plot_type="orthoview",
        plot_src_space: bool = False,
        plot_distance_to_inner_skull: bool = False,
        plot_highlighted_region: int = None,
        plot_cortical: bool = None,
        **kwargs,
    ):
        (
            src_space_points,
            region_centroid,
            pt_dist,
            pt_text,
            volume_colors,
        ) = self.prepare_plot_data_2d(
            plot_src_space,
            plot_distance_to_inner_skull,
            plot_highlighted_region,
            plot_cortical=plot_cortical,
            **kwargs,
        )
        if plot_type == "orthoview":
            fig, axs = orthoview(
                self.cerebra_volume,
                self.affine,
                src_space_points=src_space_points,
                pt_dist=pt_dist,
                plot_highlighted_region=plot_highlighted_region,
                region_centroid=region_centroid,
                pt_text=pt_text,
                volume_colors=volume_colors,
                **kwargs,
            )
            return fig, axs

        elif plot_type == "single":
            fig, ax = plot_brain_slice_2d(
                self.cerebra_volume,
                self.affine,
                src_space_points=src_space_points,
                pt_dist=pt_dist,
                plot_highlighted_region=plot_highlighted_region,
                region_centroid=region_centroid,
                pt_text=pt_text,
                volume_colors=volume_colors,
                **kwargs,
            )
            return fig, ax

        else:
            raise ValueError(f"Unknown plot_data plot_type argument {plot_type=}")

    # @time_func_decorator
    def orthoview(
        self,
        **kwargs,
    ):
        fig, axs = self.plot_data_2d(plot_type="orthoview", **kwargs)
        if "pt" in kwargs and kwargs["pt"] is not None:
            reg_id = self.get_region_id_from_point(kwargs["pt"])
            reg_name = self.get_region_name_from_region_id(reg_id)
            fig.suptitle(f"{reg_name} id ({reg_id})")

        return fig, axs

    def plot_2d(
        self,
        **kwargs,
    ):
        fig, axs = self.plot_data_2d(plot_type="single", **kwargs)
        if "pt" in kwargs and kwargs["pt"] is not None:
            reg_id = self.get_region_id_from_point(kwargs["pt"])
            reg_name = self.get_region_name_from_region_id(reg_id)
            fig.suptitle(f"{reg_name} id ({reg_id})")

        return fig, axs

    def plot_3d(
        self,
        alpha=1,
        plot_src_space: bool = False,
        plot_cortical: bool = False,
        plot_highlighted_regions: list[int] = None,
        **kwargs,
    ):


        src_space_pc = None
        if plot_src_space:
            src_space_pc = self.src_space_points

        volume_colors = None
        if plot_cortical:
            from .plotting import cortical_color, non_cortical_color
            volume_colors = [
                cortical_color if is_cortical else non_cortical_color
                for is_cortical in self.label_details["cortical"]
            ]

        highlighted_regions_pts = None
        if plot_highlighted_regions is not None:
            highlighted_regions_pts = [self.get_points_from_region_id(region_id) for region_id in plot_highlighted_regions]

        return plot_volume_3d(
            self.cerebra_volume,
            # density=6,
            alpha=alpha,
            src_space_pc=src_space_pc,
            volume_colors=volume_colors,
            highlighted_regions_pts=highlighted_regions_pts,
            **kwargs,
        )



def get_cerebra_volume(cerebra_mgz, wm_mgz):
    from .utils import get_volume_ras
    cerebra_volume, cerebra_affine = get_volume_ras(cerebra_mgz)
    wm_volume, _ = get_volume_ras(wm_mgz)

    # Add whitematter to volume data
    cerebra_volume[(wm_volume != 0) & (cerebra_volume == 0)] = 103
    return cerebra_volume, cerebra_affine


if __name__ == "__main__":
    from core.utils import setup_logging

    setup_logging(level="DEBUG")
    cerebra = CerebrA()
