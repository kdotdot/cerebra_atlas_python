import os
import logging
import nibabel as nib
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import os.path as op
import pickle

from cerebra_atlas_python.config import BaseConfig
from cerebra_atlas_python.mni_average import MNIAverage

from cerebra_atlas_python.utils import (
    download_file_from_google_drive,
    setup_logging,
    move_volume_from_LIA_to_RAS,
    expand_volume,
    find_closest_point,
    time_func_decorator,
)
from cerebra_atlas_python.plotting import (
    plot_brain_slice_2D,
    add_region_plot_to_ax,
    plot_volume_3d,
    orthoview,
    orthoview_region,
    get_3d_fig_ax,
)


# Download:
# https://drive.google.com/file/d/13rfrvxVQe18ss2hccPy10DkKQdnNyjWL/view?usp=sharing
# https://drive.google.com/file/d/1RoOfEiqglZ6wM2gU6Qae48uc3j8DXv5d/view?usp=sharing


def preprocess_label_details(df):
    # Remove first row
    df.drop(0, inplace=True)

    # Remove unused columns
    df.drop(columns=["Unnamed: 3", "Notes", "Dice Kappa"], inplace=True)

    # Change id column from string to int
    df["CerebrA ID"] = pd.to_numeric(df["CerebrA ID"])

    # Copy df and append
    df = pd.concat([df, df])
    df.reset_index(inplace=True, drop=True)

    # Modify left side labels
    df.loc["52":, "CerebrA ID"] = df.loc["52":, "CerebrA ID"] + 51

    # Modify names to include hemisphere
    df.loc["52":, "Label Name"] = "Left " + df.loc["52":, "Label Name"]
    df.loc[:"52", "Label Name"] = "Right " + df.loc[:"52", "Label Name"]

    # Add white matter to label details
    df.loc[len(df.index)] = [0, "White matter", 103]

    return df


def get_label_details(path):
    return preprocess_label_details(pd.read_csv(path))


def get_volume_RAS(path, dtype=np.uint8):
    img = nib.load(path)  # All volumes are in LIA coordinate frame
    volume, affine = move_volume_from_LIA_to_RAS(
        np.array(img.dataobj, dtype=dtype), img.affine
    )
    return volume, affine


def get_cerebra_volume(cerebra_mgz, wm_mgz):
    cerebra_volume, cerebra_affine = get_volume_RAS(cerebra_mgz)
    wm_volume, _ = get_volume_RAS(wm_mgz)

    # Add whitematter to volume data
    cerebra_volume[(wm_volume != 0) & (cerebra_volume == 0)] = 103
    return cerebra_volume, cerebra_affine


class CerebrA(BaseConfig):
    def __init__(
        self,
        mni_average=None,
        MNIAverageKwArgs=None,
        **kwargs,
    ):
        self.cerebra_output_path: str = None
        self.default_data_path: str = None
        default_config = {
            "cerebra_output_path": "./generated/cerebra",
            "default_data_path": "../cerebra_data/cerebra",
        }

        super().__init__(
            parent_name=self.__class__.__name__,
            default_config=default_config,
            **kwargs,
        )

        self.bem_surfaces = None
        self.src_volume = None

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
        self.cerebra_path = cerebra_path = op.join(
            self.default_data_path, "CerebrA_in_head.mgz"
        )
        # t1_path = op.join(self.mni_average.fs_subjects_dir, "MNIAverage/mri/T1.mgz")
        wm_path = op.join(
            self.mni_average.fs_subjects_dir, "MNIAverage/mri/wm.asegedit.mgz"
        )

        # Set volume
        self.cerebra_volume, self.affine = get_cerebra_volume(cerebra_path, wm_path)

        # Read labels
        label_details_path = op.join(self.default_data_path, "CerebrA_LabelDetails.csv")
        self.label_details = get_label_details(label_details_path)

        # Metadata
        self.region_ids = np.sort(self.label_details["CerebrA ID"].unique())

        # Sparse representation
        cerebra_sparse_path = op.join(self.cerebra_output_path, "CerebrA_sparse.npy")
        if not op.exists(cerebra_sparse_path):
            logging.info("Generating sparse representation of Cerebra volume data...")
            self.volume_data_sparse = {
                region_id: self.get_points_from_region_id(region_id)
                for region_id in self.region_ids
            }
            with open(cerebra_sparse_path, "wb") as handle:
                pickle.dump(
                    self.volume_data_sparse, handle, protocol=pickle.HIGHEST_PROTOCOL
                )
        else:
            with open(cerebra_sparse_path, "rb") as handle:
                self.volume_data_sparse = pickle.load(handle)

    # COORDINATE FRAME TRANSFORMATIONS
    def center_ras(self, pts: np.ndarray) -> np.ndarray:
        #  RAS (non-zero origin) -> RAS
        return mne.transforms.apply_trans(self.affine, pts) * 1000

    # def voxel_to_ras(self, pt):
    #     return mne.transforms.apply_trans(self.affine, pt).astype(int)

    def ras_to_voxel(self, pt):
        return np.round(
            mne.transforms.apply_trans(np.linalg.inv(self.affine), pt)
        )  # NOTE: removed .astype(int)

    def get_bem_surfaces(self):
        if self.bem_surfaces is None:
            self.bem_surfaces = self.mni_average.get_bem_surfaces_voxel_ras(
                transform=self.affine
            )
        return self.bem_surfaces  # RAS coordinate frame

    def get_src_volume(self):
        if self.src_volume is None:
            src_space_ras_nzo = self.mni_average.get_src_space_ras_nzo(
                transform=self.affine
            )
            src_space_ras = self.center_ras_volume(
                src_space_ras_nzo
            )  # Points in RAS coord frame
            # return src_volume
            src_volume = np.zeros((256, 256, 256)).astype(int)

            for i, pt in enumerate(src_space_ras):
                x, y, z = pt
                if i in self.mni_average.src["vertno"]:
                    src_volume[x, y, z] = 1  # Usable source space
                else:
                    src_volume[x, y, z] = 2  # Box around source space
            # TODO: [important] WHY WORK WITH VOXELS INSTEAD OF POINTS DIRECTLY
            self.src_volume = src_space_ras
        return self.src_volume  # voxel RAS coordinate frame

    @time_func_decorator
    def orthoview(
        self,
        plot_src_space=False,
        plot_bem_surfaces=False,
        plot_distance_to_inner_skull=False,
        **kwargs,
    ):
        src = None
        if plot_src_space:
            src = self.get_src_volume()

        bem_surfaces = None
        if plot_bem_surfaces:
            bem_surfaces = self.get_bem_surfaces()

        pt_dist = None
        if "pt" in kwargs.keys() and plot_distance_to_inner_skull:
            pt = kwargs["pt"]
            pt_dist = self.get_distance_to_inner_skull(pt)

        fig, axs = orthoview(
            self.cerebra_volume,
            self.affine,
            src_space=src,
            bem_surfaces=bem_surfaces,
            pt_dist=pt_dist,
            **kwargs,
        )

        if "pt" in kwargs.keys() and kwargs["pt"] is not None:
            reg_name = self.get_region_name_from_point(kwargs["pt"])
            reg_id = self.get_region_id_from_point(kwargs["pt"])
            fig.suptitle(f"{reg_name} id ({reg_id})")

        return axs

    def plot_region_orthoview(self, region_id, plot_src_space=False, **kwargs):
        src = None
        if plot_src_space:
            src = self.mni_average.get_src_volume(transform=self.affine)
        reg_name = self.get_region_name_from_region_id(region_id)
        reg_points = self.get_points_from_region_id(region_id)
        reg_centroid = self.find_region_centroid_from_name(reg_name)

        fig, axs = orthoview_region(
            reg_points,
            reg_centroid,
            volume=self.cerebra_volume,
            affine=self.affine,
            region_id=region_id,
            src_space=src,
            **kwargs,
        )

        fig.suptitle(reg_name)

        return axs

    def plot_3d(self, alpha=1):
        return plot_volume_3d(self.cerebra_volume, density=6, alpha=alpha)

    def plot_region_3d(
        self,
        region_id,
    ):
        pts = self.get_points_from_region_id(region_id)
        plot_volume_3d(self.cerebra_volume, region_pts=pts, density=6)

    def plot_whitematter_3d(self, **kwargs):
        pts = self.get_points_from_region_id(103)
        fig, ax = plot_volume_3d(self.cerebra_volume, density=6, alpha=0.5)
        _, ax = plot_volume_3d(
            self.cerebra_volume, region_pts=pts, density=64, ax=ax, **kwargs
        )

    # Given a point in a (256,256,256) 3d space (RAS), determines
    # which brain region it belongs to
    def get_region_name_from_point(self, point):
        label_id = self.cerebra_volume[point[0], point[1], point[2]]
        if label_id != 0:
            return self.label_details[self.label_details["CerebrA ID"] == label_id][
                "Label Name"
            ].item()

    def get_region_id_from_point(self, point):
        label_id = self.cerebra_volume[point[0], point[1], point[2]]
        return int(label_id)  # Can be 0 and 103

    # Given a brain region name, get an array of points that
    # make up said region
    def get_points_from_region_name(self, region_name):
        region_data = self.label_details[
            self.label_details["Label Name"] == region_name
        ]
        if len(region_data) < 1:
            return None
        return self.get_points_from_region_id(region_data["CerebrA ID"].item())

    # Helper function for get_points_from_region_name
    # Does the same but with region id instead of region name
    def get_points_from_region_id(self, region_id):
        return self.volume_data_sparse[region_id]

    def get_region_name_from_region_id(self, region_id):
        if region_id == 0:
            return ""
        if region_id > 103:
            return "Non-valid id"
        return self.label_details[self.label_details["CerebrA ID"] == region_id][
            "Label Name"
        ].item()

    def get_distance_to_inner_skull(self, pt):
        _, _, inner_skull_points = self.get_bem_surfaces()
        closest_point, distance = find_closest_point(inner_skull_points, pt)

        return closest_point, distance

    def get_closest_region_to_whitematter(self, pt, n_max=20):
        success = False

        region_id = self.get_region_id_from_point(pt)
        if region_id != 103 and region_id != 0:
            logging.warning(
                f"Attempting to get closest region to whitematter from a non-whitematter region -> {region_id= }"
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
            f"get_closest_region_to_whitematter unable to find close region {n_max= }"
        )
        return success, pt, region_id

    def find_region_centroid_from_name(self, region_name):
        region_data = self.label_details[
            self.label_details["Label Name"] == region_name
        ]
        if len(region_data) < 1:
            return None
        return self.find_region_centroid_from_id(region_data["CerebrA ID"].item())

    def find_region_centroid_from_id(self, region_id):
        return np.round(self.get_points_from_region_id(region_id).mean(axis=0)).astype(
            int
        )


if __name__ == "__main__":
    setup_logging()
    cerebra = CerebrA()

    print(cerebra.volume_data.min(), cerebra.volume_data.max())
