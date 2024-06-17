"""
Cerebra...
"""

import os.path as op
import logging
from typing import Dict, List
import numpy as np
import mne
import pandas as pd
import matplotlib

from .plotting import (
    plot_volume_3d,
    orthoview,
    plot_brain_slice_2d,
    get_cmap_colors_hex,
)
from .utils import point_cloud_to_voxel
from .config import Config
from .__cache import cache_pkl, cache_np, cache_mne_src
from .cerebra_base import CerebraBase
from .transforms import lia_to_ras




class CerebrA(CerebraBase, Config):
    def __init__(
        self,
        **kwargs,
    ):






    ######
    # * METHODS

    def get_region_id_from_point(self, point):
        point = np.array(point).astype(int)
        region_id = self.cerebra_volume[point[0], point[1], point[2]]
        return int(region_id)  # Can be get_points_from_region_id0 and 103

    # Given a point in a (256,256,256) 3d space (RAS), determines
    # which brain region it belongs to
    def get_region_name_from_point(self, point):
        region_id = self.get_region_id_from_point(point)
        region_name = self.get_region_name_from_region_id(region_id)
        return region_name

 

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

    def get_visual_cortex_region_ids(self):

        perception_visual_ambient = np.array([9, 31, 60, 82])

        return perception_visual_ambient

    def get_cortical_id_from_region_id(self, region_id):
        return (np.where(self.get_cortical_region_ids() == region_id)[0][0]) + 1

    def get_cortical_colors(self, rgba=False, rgb=False):
        if rgba and rgb:
            raise ValueError("Only one of {rgba,rgb} should be True")
        colors = get_cmap_colors_hex()
        if rgb:
            colors = [matplotlib.colors.to_rgb(c) for c in colors]
        if rgba:
            colors = [matplotlib.colors.to_rgba(c) for c in colors]
        cortical_colors = [
            colors[region_id] for region_id in self.get_cortical_region_ids()
        ]
        return cortical_colors

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
        plot_highlighted_regions: int = None,
        plot_cortical: bool = None,
        plot_cortical_ids: bool = None,
        **kwargs,
    ):

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
                (
                    matplotlib.colors.to_rgb(cortical_color)
                    if c
                    else matplotlib.colors.to_rgb(non_cortical_color)
                )
                for c in cortical_mask
            ]
            volume_colors = np.array(volume_colors)

        (
            highlighted_region_ids,
            highlighted_region_names,
            highlighted_region_centroids,
        ) = (None, None, None)
        if plot_highlighted_regions is not None:
            highlighted_region_ids = np.array(plot_highlighted_regions)
            highlighted_region_names = [
                self.get_region_name_from_region_id(i) for i in plot_highlighted_regions
            ]
            if plot_cortical_ids:
                highlighted_cortical_ids = [
                    self.get_cortical_id_from_region_id(i)
                    for i in plot_highlighted_regions
                ]
            highlighted_region_centroids = [
                self.find_region_centroid_from_id(i) for i in plot_highlighted_regions
            ]

        ret_data = {
            "src_space_points": self.src_space_points if plot_src_space else None,
            "highlighted_region_ids": highlighted_region_ids,
            "highlighted_region_names": highlighted_region_names,
            "highlighted_region_centroids": highlighted_region_centroids,
        }
        if plot_cortical_ids:
            ret_data["highlighted_cortical_ids"] = highlighted_cortical_ids
        return ret_data
        # (
        #     src_space_points,
        #     region_centroid,
        #     pt_dist,
        #     pt_text,
        #     volume_colors,
        #     t1_volume,
        # )

    def plot_data_2d(
        self,
        plot_type="orthoview",
        plot_src_space: bool = False,
        plot_distance_to_inner_skull: bool = False,
        plot_highlighted_regions: List[int] = None,
        plot_cortical_ids: bool = None,
        **kwargs,
    ):

        plot_data = self.prepare_plot_data_2d(
            plot_src_space=plot_src_space,
            plot_distance_to_inner_skull=plot_distance_to_inner_skull,
            plot_highlighted_regions=plot_highlighted_regions,
            plot_cortical_ids=plot_cortical_ids,
            **kwargs,
        )
        if plot_type == "orthoview":
            fig, axs = orthoview(
                self.cerebra_volume,
                self.affine,
                **plot_data,
                **kwargs,
            )
            return fig, axs

        elif plot_type == "single":
            fig, ax = plot_brain_slice_2d(
                self.cerebra_volume,
                self.affine,
                **plot_data,
                **kwargs,
            )
            return fig, ax

        else:
            raise ValueError(f"Unknown {plot_type=}")

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
        highlighted_regions_alphas: int = 1,
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
            downsample_factor = 8
            highlighted_regions_pts = [
                self.get_points_from_region_id(region_id)[::downsample_factor]
                for region_id in plot_highlighted_regions
            ]

        return plot_volume_3d(
            self.cerebra_volume,
            # density=6,
            alpha=alpha,
            src_space_pc=src_space_pc,
            volume_colors=volume_colors,
            highlighted_regions_pts=highlighted_regions_pts,
            highlighted_regions_alphas=highlighted_regions_alphas,
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
