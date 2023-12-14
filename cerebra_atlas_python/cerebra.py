"""
Cerebra...
"""
import os
import os.path as op
import logging
from typing import Dict
import pickle
import numpy as np
import pandas as pd
import mne
import matplotlib


from .plotting import (
    plot_volume_3d,
    orthoview,
    plot_brain_slice_2d,
    get_cmap_colors_hex,
)
from .config import BaseConfig
from .mni_average import MNIAverage
from .utils import (
    move_volume_from_lia_to_ras,
    find_closest_point,
    merge_voxel_grids,
    point_cloud_to_voxel,
    move_volume_from_ras_to_lia,
    get_volume_ras,
)


class CerebrA(BaseConfig):
    def __init__(
        self,
        config_path=op.dirname(__file__) + "/config.ini",
        mni_average=None,
        MNIAverageKwArgs=None,
        **kwargs,
    ):
        self.cerebra_output_path: str = None
        self.cerebra_data_path: str = None
        self.source_space_grid_size: int = None
        self.source_space_include_wm: bool = None
        self.source_space_include_non_cortical = None
        default_config = {
            "cerebra_output_path": "./generated/cerebra",
            "cerebra_data_path": op.dirname(__file__) + "/cerebra_data",
            "source_space_grid_size": 3,  # mm
            "source_space_include_wm": False,
            "source_space_include_non_cortical": True,
        }

        super().__init__(
            parent_name=self.__class__.__name__,
            default_config=default_config,
            config_path=config_path,
            **kwargs,
        )

        # Always load (fast/required)
        self.label_details: pd.DataFrame = None
        self.cerebra_volume: np.ndarray = None
        self.affine: np.ndarray = None
        # Load on demand [using @property] (slow)
        self._cerebra_sparse: Dict[np.ndarray] = None
        self._src_space: mne.SourceSpaces = None
        self._src_space_points: np.ndarray = None
        self._src_space_labels: np.ndarray = None
        self._src_space_n_points_per_region: np.ndarray = None
        self._src_space_n_total_points: int = None
        self._src_space_mask: np.ndarray = None
        self._bem_surfaces: np.ndarray = None
        self._bem_volume: np.ndarray = None

        # If output folder does not exist, create it
        if not op.exists(self.cerebra_output_path):
            os.makedirs(self.cerebra_output_path, exist_ok=True)

        # Instantiate/ assign MNIAverage object
        if mni_average is None:
            MNIAverageKwArgs = MNIAverageKwArgs or {}
            self.mni_average = MNIAverage(**MNIAverageKwArgs)
        else:
            assert isinstance(
                mni_average, MNIAverage
            ), f"Wrong class should be MNIAverage {type(mni_average)= }"
            self.mni_average = mni_average

        # Input paths
        cerebra_volume_path = op.join(self.cerebra_data_path, "volume.npy")
        cerebra_affine_path = op.join(self.cerebra_data_path, "affine.npy")
        label_details_path = op.join(self.cerebra_data_path, "label_details.csv")

        # Output paths
        self._cerebra_sparse_path = op.join(
            self.cerebra_output_path, "CerebrA_sparse.npy"
        )
        self._src_space_path = op.join(
            self.cerebra_output_path, f"{self.src_space_string}_src.fif"
        )
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
        self.cortical_color = "#9EC8B9"
        self.non_cortical_color = "#1B4242"

    # * PROPERTIES
    @property
    def src_space_string(self):
        return f"src_space_{self.source_space_grid_size}mm{'wm' if self.source_space_include_wm else ''}{'_nc' if self.source_space_include_non_cortical else ''}"

    @property
    def cerebra_sparse(self):
        if self._cerebra_sparse is None:
            self._set_cerebra_sparse()
        return self._cerebra_sparse

    @property
    def src_space(self):
        if self._src_space is None:
            self._set_src_space()
        return self._src_space

    @property
    def src_space_mask(self):
        if self._src_space_mask is None:
            self._set_src_space_mask()
        return self._src_space_mask

    @property
    def src_space_points(self):
        if self._src_space_points is None:
            self._set_src_space_points()
        return self._src_space_points

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

    # * SETTERS
    def _set_cerebra_sparse(self):
        if not op.exists(self._cerebra_sparse_path):
            logging.info("Generating sparse representation of Cerebra volume data...")
            self._cerebra_sparse = {
                region_id: self.calculate_points_from_region_id(region_id)
                for region_id in self.region_ids
            }
            with open(self._cerebra_sparse_path, "wb") as handle:
                pickle.dump(
                    self._cerebra_sparse, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
        else:
            with open(self._cerebra_sparse_path, "rb") as handle:
                self._cerebra_sparse = pickle.load(handle)

    def _set_src_space(self):
        if not op.exists(self._src_space_path):
            logging.info("Generating new source space %s {self._src_space_path}")
            normals = np.repeat([[0, 0, 1]], self.src_space_n_total_points, axis=0)

            rr = point_cloud_to_voxel(self.src_space_points)
            rr = move_volume_from_ras_to_lia(rr)
            rr = np.argwhere(rr != 0)  # Back to point cloud
            inv_aff = np.linalg.inv(self.mni_average.t1.affine)
            # Translation
            inv_aff[:, 3][2] = 132
            # Rotation
            inv_aff[:, 1][2] *= -1
            inv_aff[:, 2][1] *= -1
            inv_aff[:, 3][1] = -128
            rr = mne.transforms.apply_trans(inv_aff, rr)
            rr = rr / 1000
            pos = dict(rr=rr, nn=normals)
            self._src_space = mne.setup_volume_source_space(pos=pos)
            self._src_space.save(self._src_space_path, overwrite=True, verbose=False)
        else:
            logging.info("Loading source space from disk | %s ", self._src_space_path)
            self._src_space = mne.read_source_spaces(self._src_space_path)

    def _set_src_space_mask(self):
        if self._src_space_mask is None:
            if not op.exists(self._src_space_mask_path):
                self._src_space_mask = self._get_src_space_mask()
                np.save(self._src_space_mask_path, self._src_space_mask)
            else:
                self._src_space_mask = np.load(self._src_space_mask_path)
        return self._src_space_mask

    def _set_src_space_points(self):
        if self._src_space_points is None:
            if not op.exists(self._src_space_points_path):
                self._src_space_points = np.indices([256, 256, 256])[
                    :, self.src_space_mask
                ].T
                np.save(self._src_space_points_path, self._src_space_points)
            else:
                self._src_space_points = np.load(self._src_space_points_path)
        return self._src_space_points

    def _set_bem_surfaces(self):
        self._bem_surfaces = self.mni_average.get_bem_surfaces_ras_nzo(
            transform=self.affine
        )

    def _set_bem_volume(self):
        self._bem_volume = None
        for surf, bem_id in zip(self.bem_surfaces, self.mni_average.bem_names.keys()):
            if self._bem_volume is None:
                self._bem_volume = point_cloud_to_voxel(surf, vox_value=bem_id)
            else:
                self._bem_volume = merge_voxel_grids(
                    self.bem_volume, point_cloud_to_voxel(surf, vox_value=bem_id)
                )

    # * TRANSFORMS
    ######
    # * METHODS
    def _get_src_space_mask(self):
        # Create grid (downsample available volume)
        size = self.source_space_grid_size
        grid = np.zeros((256, 256, 256))
        a = np.arange(size - 1, 256, size)
        for x in a:
            for y in a:
                grid[x, y, a] = 1
        grid_mask = grid.astype(bool)

        # If include non-cortical keep all
        if self.source_space_include_non_cortical:
            not_zero_mask = self.cerebra_volume != self.get_region_id_from_region_name(
                "Empty"
            )
            not_wm_mask = self.cerebra_volume != self.get_region_id_from_region_name(
                "White matter"
            )
            downsampled_not_zero_mask = np.logical_and(grid_mask, not_zero_mask)
            downsampled_not_wm_mask = np.logical_and(grid_mask, not_wm_mask)
            # Combined mask is not zero and not wm
            combined_mask = np.logical_and(
                downsampled_not_zero_mask, downsampled_not_wm_mask
            )

        else:  # Keep only cortical:
            # Get cortical region ids
            cortical_ids = self.label_details[self.label_details["cortical"] == True][
                "CerebrA ID"
            ].values
            combined_mask = np.zeros((256, 256, 256)).astype(bool)
            for c_id in cortical_ids:
                # Handle each region individually
                region_mask = self.cerebra_volume == c_id
                # Downsample based on grid
                downsampled_region_mask = np.logical_and(grid_mask, region_mask)
                # Add region to mask
                combined_mask = np.logical_or(combined_mask, downsampled_region_mask)

        # Add whitematter if needed
        if self.source_space_include_wm:
            whitematter_mask = (
                self.cerebra_volume
                == self.get_region_id_from_region_name("White matter")
            )
            # Downsample
            downsampled_whitematter_mask = np.logical_and(grid_mask, whitematter_mask)
            combined_mask = np.logical_or(combined_mask, downsampled_whitematter_mask)
        return combined_mask

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
        # TODO: see mne.bem.distance_to_bem
        _, _, inner_skull_points = self.bem_surfaces
        closest_point, distance = find_closest_point(inner_skull_points, pt)

        return closest_point, distance

    def get_closest_region_to_whitematter(self, pt, n_max=20):
        success = False

        region_id = self.get_region_id_from_point(pt)
        if region_id != 103 and region_id != 0:
            logging.warning(
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

        logging.warning(
            "Get_closest_region_to_whitematter unable to find close region %s", n_max
        )
        return success, pt, region_id

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
                logging.error(
                    "Unable to find closest region to whitematter from region centroid"
                )
            elif closest_region_id != region_id:
                logging.warning("Region centroid is outside of region?")
            centroid = pt
        return centroid

    # * PLOTTING
    def prepare_plot_data_2d(
        self,
        plot_src_space: bool = False,
        plot_bem_surfaces: bool = False,
        plot_distance_to_inner_skull: bool = False,
        plot_highlighted_region: int = None,
        plot_cortical: bool = None,
        plot_t1_volume: bool = None,
        **kwargs,
    ):
        src_space_points = None
        if plot_src_space:
            src_space_points = self.src_space_points

        bem_volume = None
        if plot_bem_surfaces:
            bem_volume = self.bem_volume

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
            volume_colors = [
                matplotlib.colors.to_rgb(self.cortical_color)
                if c
                else matplotlib.colors.to_rgb(self.non_cortical_color)
                for c in cortical_mask
            ]
            volume_colors = np.array(volume_colors)

        t1_volume = None
        if plot_t1_volume:
            t1_volume = move_volume_from_lia_to_ras(self.mni_average.t1.dataobj)

        return (
            src_space_points,
            bem_volume,
            region_centroid,
            pt_dist,
            pt_text,
            volume_colors,
            t1_volume,
        )

    def plot_data_2d(
        self,
        plot_type="orthoview",
        plot_src_space: bool = False,
        plot_bem_surfaces: bool = False,
        plot_distance_to_inner_skull: bool = False,
        plot_highlighted_region: int = None,
        plot_cortical: bool = None,
        plot_t1_volume: bool = None,
        **kwargs,
    ):
        (
            src_space_points,
            bem_volume,
            region_centroid,
            pt_dist,
            pt_text,
            volume_colors,
            t1_volume,
        ) = self.prepare_plot_data_2d(
            plot_src_space,
            plot_bem_surfaces,
            plot_distance_to_inner_skull,
            plot_highlighted_region,
            plot_cortical=plot_cortical,
            plot_t1_volume=plot_t1_volume,
            **kwargs,
        )
        if plot_type == "orthoview":
            fig, axs = orthoview(
                self.cerebra_volume,
                self.affine,
                src_space_points=src_space_points,
                bem_volume=bem_volume,
                pt_dist=pt_dist,
                plot_highlighted_region=plot_highlighted_region,
                region_centroid=region_centroid,
                pt_text=pt_text,
                volume_colors=volume_colors,
                t1_volume=t1_volume,
                **kwargs,
            )
            return fig, axs

        elif plot_type == "single":
            fig, ax = plot_brain_slice_2d(
                self.cerebra_volume,
                self.affine,
                src_space_points=src_space_points,
                bem_volume=bem_volume,
                pt_dist=pt_dist,
                plot_highlighted_region=plot_highlighted_region,
                region_centroid=region_centroid,
                pt_text=pt_text,
                volume_colors=volume_colors,
                t1_volume=t1_volume,
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
        plot_bem=False,
        plot_src_space: bool = False,
        plot_cortical: bool = False,
        plot_highlighted_region: int = None,
        **kwargs,
    ):
        bem_surfaces = None
        if plot_bem:
            bem_surfaces = self.bem_surfaces

        src_space_pc = None
        if plot_src_space:
            src_space_pc = self.src_space_points

        volume_colors = None
        if plot_cortical:
            volume_colors = [
                self.cortical_color if is_cortical else self.non_cortical_color
                for is_cortical in self.label_details["cortical"]
            ]

        region_pts = None
        if plot_highlighted_region is not None:
            region_pts = self.get_points_from_region_id(plot_highlighted_region)

        return plot_volume_3d(
            self.cerebra_volume,
            # density=6,
            alpha=alpha,
            bem_surfaces=bem_surfaces,
            src_space_pc=src_space_pc,
            volume_colors=volume_colors,
            region_pts=region_pts,
            **kwargs,
        )


def preprocess_label_details(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the given dataframe by performing several operations such as removing rows and columns,
    converting data types, duplicating and modifying data, and appending new information.

    Args:
        df (pd.DataFrame): The dataframe to preprocess.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    # Remove first row
    df.drop(0, inplace=True)

    # Remove unused columns
    df.drop(columns=["Unnamed: 3", "Notes", "Dice Kappa"], inplace=True)

    # Change id column from string to int
    df["CerebrA ID"] = pd.to_numeric(df["CerebrA ID"])
    df["CerebrA ID"] = df["CerebrA ID"].astype("uint8")

    # Copy df and append
    df = pd.concat([df, df])
    df.reset_index(inplace=True, drop=True)

    # Modify left side labels
    df.loc["51":, "CerebrA ID"] = df.loc["51":, "CerebrA ID"] + 51

    # df["Mindboggle ID"] = df["Mindboggle ID"].astype("uint16")

    # Modify names to include hemisphere
    df["hemisphere"] = ""
    # df.loc[:, "hemisphere"] = 12
    df.loc["51":, "hemisphere"] = "Left"
    df.loc[:"50", "hemisphere"] = "Right"

    # Label cortical regions
    df["cortical"] = df["Mindboggle ID"] > 1000

    # Adjust Mindboggle ids
    # (see https://mindboggle.readthedocs.io/en/latest/labels.html)
    mask = df["cortical"] & (df["hemisphere"] == "Left")
    df.loc[mask, "Mindboggle ID"] = df.loc[mask, "Mindboggle ID"] - 1000

    # Add white matter to label details
    df.loc[len(df.index)] = [0, "White matter", 103, "", False]

    # Add 'empty' to label details
    df.loc[len(df.index)] = [0, "Empty", 0, "", False]

    df.sort_values(by=["CerebrA ID"], inplace=True)
    df.reset_index(inplace=True, drop=True)

    # Add hemispheres

    # Add colors
    # Order by CerebrA ID then get colors
    df["color"] = get_cmap_colors_hex()

    return df


def get_label_details(path):
    """Reads a CSV file from the given path and preprocesses its contents using the preprocess_label_details function.
    Returns:
        pd.DataFrame: The preprocessed dataframe obtained from the CSV file.
    """
    return preprocess_label_details(pd.read_csv(path))


def get_cerebra_volume(cerebra_mgz, wm_mgz):
    """
    Processes cerebra and white matter medical image volumes to integrate
    white matter information into the cerebra volume.

    This function:
    1. Retrieves the volume data and affine matrices for both cerebra and white matter images in RAS coordinate frame.
    2. Modifies the cerebra volume by adding a specific label (103) to represent white matter regions
       that are not already labeled in the cerebra volume.

    Args:
        cerebra_mgz (str): The file path of the cerebra medical image volume.
        wm_mgz (str): The file path of the white matter medical image volume.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the modified cerebra volume data and its affine matrix.
    """
    cerebra_volume, cerebra_affine = get_volume_ras(cerebra_mgz)
    wm_volume, _ = get_volume_ras(wm_mgz)

    # Add whitematter to volume data
    cerebra_volume[(wm_volume != 0) & (cerebra_volume == 0)] = 103
    return cerebra_volume, cerebra_affine


if __name__ == "__main__":
    from .utils import setup_logging

    setup_logging(level="DEBUG")
    cerebra = CerebrA()
