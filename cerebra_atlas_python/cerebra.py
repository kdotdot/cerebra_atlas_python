"""
Cerebra...
"""
import os
import os.path as op
import logging
import pickle
import nibabel as nib
import numpy as np
import pandas as pd
import mne

from .plotting import (
    plot_volume_3d,
    orthoview,
    plot_brain_slice_2d,
    get_cmap_colors_hex,
)
from .config import BaseConfig
from .mni_average import MNIAverage
from .utils import (
    setup_logging,
    move_volume_from_lia_to_ras,
    find_closest_point,
    merge_voxel_grids,
    point_cloud_to_voxel,
    move_volume_from_ras_to_lia,
)


def get_volume_ras(path, dtype=np.uint8):
    """
    Loads a medical image volume from the given path and converts its coordinate frame from LIA to RAS.

    This function:
    1. Loads the volume using nibabel.
    2. Converts the volume's coordinate frame from LIA (Left, Inferior, Anterior) to RAS (Right, Anterior, Superior)
       using the move_volume_from_lia_to_ras function.

    Args:
        path (str): The file path of the medical image volume.
        dtype (type, optional): The data type to be used for the volume data. Defaults to np.uint8.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the transformed volume data and its affine matrix.
    """
    img = nib.load(path)  # All volumes are in LIA coordinate frame
    volume, affine = move_volume_from_lia_to_ras(
        np.array(img.dataobj, dtype=dtype), img.affine
    )
    return volume, affine


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
            "remove_empty_from_src_space": True,
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

        # self.bem_surfaces = None  # Array of 3 bem point clouds
        # self.bem_volume = (
        #     None  # Voxel grid of size [256, 256, 256] : 1 represent source space
        # )
        # self.src_space_pc = None  # Source space point cloud
        # self.src_space_volume = (
        #     None  # Voxel grid of size [256, 256, 256] : 1 represent source space
        # )

        # Instantiate/ assign MNIAverage object
        if mni_average is None:
            MNIAverageKwArgs = MNIAverageKwArgs or {}
            self.mni_average = MNIAverage(**MNIAverageKwArgs)
        else:
            assert isinstance(
                mni_average, MNIAverage
            ), f"Wrong class should be MNIAverage {type(mni_average)= }"

            self.mni_average = mni_average

        # If output folder does not exist, create it
        if not op.exists(self.cerebra_output_path):
            os.makedirs(self.cerebra_output_path, exist_ok=True)

        # Define volumes' path
        self.cerebra_in_head_path = op.join(
            self.cerebra_data_path, "CerebrA_in_head.mgz"
        )

        # Set volume
        self.cerebra_volume, self.affine = get_cerebra_volume(
            self.cerebra_in_head_path, self.mni_average.wm_path
        )

        # Read labels
        label_details_path = op.join(self.cerebra_data_path, "label_details.csv")
        self.label_details = pd.read_csv(label_details_path)

        # Metadata
        self.region_ids = np.sort(self.label_details["CerebrA ID"].unique())

        # Sparse representation
        cerebra_sparse_path = op.join(self.cerebra_output_path, "CerebrA_sparse.npy")
        if not op.exists(cerebra_sparse_path):
            logging.info("Generating sparse representation of Cerebra volume data...")
            self.volume_data_sparse = {
                region_id: self.calculate_points_from_region_id(region_id)
                for region_id in self.region_ids
            }
            with open(cerebra_sparse_path, "wb") as handle:
                pickle.dump(
                    self.volume_data_sparse, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
        else:
            with open(cerebra_sparse_path, "rb") as handle:
                self.volume_data_sparse = pickle.load(handle)

        # Constant attributes
        self.cortical_color = "#9EC8B9"
        self.non_cortical_color = "#1B4242"

        self.src_space_labels = None
        self.src_space = None
        self.src_space_points = None
        self._set_src_space()

    @property
    def bem_names(self):
        return self.mni_average.bem_names

    # COORDINATE FRAME TRANSFORMATIONS

    def src_vertex_index_to_ras_voxel(self, vert_id: np.ndarray) -> np.ndarray:
        pts = self.mni_average.src_vertex_index_to_mri(vert_id)
        pts = self.mni_average.mri_to_ras_nzo(pts)
        #  RAS (non-zero origin) -> RAS
        return np.squeeze(self.center_ras(pts)).astype(int)

    def center_ras(self, pts: np.ndarray) -> np.ndarray:
        #  RAS (non-zero origin) -> RAS
        return mne.transforms.apply_trans(self.affine, pts)

    def inverse_center_ras(self, pts: np.ndarray) -> np.ndarray:
        return mne.transforms.apply_trans(np.linalg.inv(self.affine), pts)

    # CENTERED MNI_AVERAGE POINTS
    # Returns array of length 3 of point clouds
    def get_bem_surfaces(self):
        if self.bem_surfaces is None:
            self.bem_surfaces = self.mni_average.get_bem_surfaces_ras_nzo(
                transform=self.affine
            )
        return self.bem_surfaces  # RAS coordinate frame

    def get_bem_volume(self):
        if self.bem_volume is None:
            bem_surfaces = self.get_bem_surfaces()
            bem_volume = None
            for surf, bem_id in zip(bem_surfaces, self.mni_average.bem_names.keys()):
                if bem_volume is None:
                    bem_volume = point_cloud_to_voxel(surf, vox_value=bem_id)
                else:
                    bem_volume = merge_voxel_grids(
                        bem_volume, point_cloud_to_voxel(surf, vox_value=bem_id)
                    )
            self.bem_volume = bem_volume
        return self.bem_volume

    # If vol src fif does not exist, create it, otherwise read it
    def _set_src_space(self):
        """Internal method to set up the volume source space."""
        logging.debug("Generating discrete source space...")
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

        self.src_space_labels = self.cerebra_volume[combined_mask]
        self.src_space_points = np.indices([256, 256, 256])[:, combined_mask].T
        normals = np.repeat([[0, 0, 1]], len(self.src_space_points), axis=0)
        #
        # TODO: Transform to original coordinate frame
        # rr = self.inverse_center_ras(self.src_space_points)
        rr = point_cloud_to_voxel(self.src_space_points)
        rr = move_volume_from_ras_to_lia(rr)
        rr = np.argwhere(rr != 0)  # Back to point cloud

        inv_aff = np.linalg.inv(self.mni_average.t1.affine)

        # Translation
        inv_aff[:, 3][2] = 132
        #
        # inv_aff[:, 3][3] = 178

        # Rotation
        inv_aff[:, 1][2] *= -1
        inv_aff[:, 2][1] *= -1
        # inv_aff[:, 3][3] *= -1
        inv_aff[:, 3][1] = -128

        # rr = self.mni_average.inverse_center_lia(rr)
        rr = mne.transforms.apply_trans(inv_aff, rr)

        rr = rr / 1000

        pos = dict(rr=rr, nn=normals)
        self.src_space = mne.setup_volume_source_space(pos=pos)
        # self.src_space.save(self._src_space_path, overwrite=True, verbose=True)

    # Returns point cloud
    def get_src_space_pc(self):
        return self.src_space_points

    def get_vol_src_pc(
        self,
        include_non_cortical=True,
        include_whitematter=False,
        hemisphere=None,  # NOTE: include_regions could be added
    ):
        if hemisphere is not None and hemisphere not in ["Left", "Right"]:
            raise ValueError(
                f"Invalid hemisphere {hemisphere=} should be [Left, Right]"
            )

        cortical = self.label_details[self.label_details["cortical"] == True]
        non_cortical = self.label_details[self.label_details["cortical"] == False]
        whitematter = self.label_details[
            self.label_details["Label Name"] == "White matter"
        ]
        empty = self.label_details[self.label_details["Label Name"] == "Empty"]

        if include_non_cortical:
            all_regions = pd.concat([cortical, non_cortical, whitematter])
        else:
            all_regions = pd.concat([cortical, whitematter])

        # Drop 'Empty' region -> id 0
        all_regions.drop(empty["CerebrA ID"], inplace=True, errors="ignore")
        # Remove whitematter if needed
        if not include_whitematter:
            all_regions.drop(whitematter["CerebrA ID"], inplace=True, errors="ignore")
        # Remove hemisphere if needed
        if hemisphere is not None:
            drop_hemisphere = "Left" if hemisphere == "Right" else "Right"
            all_regions.drop(
                all_regions[all_regions["hemisphere"] == drop_hemisphere]["CerebrA ID"],
                inplace=True,
            )

        ids = all_regions["CerebrA ID"].values

        print(ids)

        return ids

    def get_src_space_volume(self):
        if self.src_space_volume is None:
            src_space_pc = self.get_src_space_pc()
            self.src_space_volume = point_cloud_to_voxel(src_space_pc, vox_value=1)
        return self.src_space_volume

    def modify_src_space(self, src_space):
        new_inuse = []
        for i, is_inuse in enumerate(src_space[0]["inuse"]):
            # Do not modify not in use
            if not is_inuse:
                new_inuse.append(0)
                continue
            # If used, check whether it falls outside the brain
            pt = self.src_vertex_index_to_ras_voxel(i)
            region_id = self.get_region_id_from_point(pt)
            if region_id != 0:
                new_inuse.append(1)
            else:
                new_inuse.append(0)
        src_space[0]["inuse"] = np.array(new_inuse)
        src_space[0]["vertno"] = np.where(src_space[0]["inuse"])[0]
        src_space[0]["nuse"] = len(src_space[0]["vertno"])
        src_space[0]["neighbor_vert"] = get_neighbors(src_space[0])

        return src_space

    # Functions
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
        return np.array(np.where(self.cerebra_volume == region_id)).T

    def get_points_from_region_id(self, region_id):
        return self.volume_data_sparse[region_id]

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
        _, _, inner_skull_points = self.get_bem_surfaces()
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

    def prepare_plot_data_2d(
        self,
        plot_src_space: bool = False,
        plot_bem_surfaces: bool = False,
        plot_distance_to_inner_skull: bool = False,
        plot_highlighted_region: int = None,
        **kwargs,
    ):
        src_space_points = None
        if plot_src_space:
            src_space_points = self.src_space_points

        bem_volume = None
        if plot_bem_surfaces:
            bem_volume = self.get_bem_volume()

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

        return src_space_points, bem_volume, region_centroid, pt_dist, pt_text

    def plot_data_2d(
        self,
        plot_type="orthoview",
        plot_src_space: bool = False,
        plot_bem_surfaces: bool = False,
        plot_distance_to_inner_skull: bool = False,
        plot_highlighted_region: int = None,
        **kwargs,
    ):
        (
            src_space_points,
            bem_volume,
            region_centroid,
            pt_dist,
            pt_text,
        ) = self.prepare_plot_data_2d(
            plot_src_space,
            plot_bem_surfaces,
            plot_distance_to_inner_skull,
            plot_highlighted_region,
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
            bem_surfaces = self.get_bem_surfaces()

        src_space_pc = None
        if plot_src_space:
            src_space_pc = self.get_src_space_pc()

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


if __name__ == "__main__":
    setup_logging()
    cerebra = CerebrA(source_space_include_non_cortical=False)
