import os
import logging
import nibabel as nib
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from cerebra_atlas_python.utils import (
    download_file_from_google_drive,
    setup_logging,
    move_volume_from_LIA_to_RAS,
)
from cerebra_atlas_python.plotting import (
    plot_brain_slice_2D,
    add_region_plot_to_ax,
    plot_volume_3d,
    orthoview,
    orthoview_region,
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

    return df


class CerebrA:
    def __init__(self, download_dir="./data", download_data=True):
        # Define paths for required data
        # cerebra_in_head_path = f"{download_dir}/CerebrA_in_head.mgz"
        label_details_path = f"{download_dir}/CerebrA_LabelDetails.csv"

        # TODO: move to GDrive
        cerebra_in_head_path = (
            "/home/carlos/Datasets/Cerebra/10.12751_g-node.be5e62/CerebrA_in_head.mgz"
        )
        t1_path = "/home/carlos/Datasets/subjects/MNIAverage/mri/T1.mgz"
        brain_path = "/home/carlos/Datasets/subjects/MNIAverage/mri/brain.mgz"  # TODO: use wm.mgz

        # Create download folder if it does not exist
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download data if it is not present within download folder
        # TODO: Modify volumes first and then make users download volume and labels only
        if download_data and not os.path.exists(cerebra_in_head_path):
            logging.info("Downloading CerebrA volume...")
            file_id = "13rfrvxVQe18ss2hccPy10DkKQdnNyjWL"
            download_file_from_google_drive(file_id, cerebra_in_head_path)
        if download_data and not os.path.exists(label_details_path):
            logging.info("Downloading CerebrA labels...")
            file_id = "1RoOfEiqglZ6wM2gU6Qae48uc3j8DXv5d"
            download_file_from_google_drive(file_id, label_details_path)

        # Read data
        self.label_details = preprocess_label_details(pd.read_csv(label_details_path))

        cerebra_in_head_img = nib.load(cerebra_in_head_path)  # LIA coordinate frame
        self.cerebra_volume, self.affine = move_volume_from_LIA_to_RAS(
            np.array(cerebra_in_head_img.dataobj), cerebra_in_head_img.affine
        )

        brain_img = nib.load(brain_path)  # LIA coordinate frame
        self.brain_volume = move_volume_from_LIA_to_RAS(
            np.array(brain_img.dataobj)
        )  # Affine is shared with cerebra in head

        t1_img = nib.load(t1_path)  # LIA coordinate frame
        self.t1_volume = move_volume_from_LIA_to_RAS(np.array(t1_img.dataobj))

        # Add whitematter to volume data
        self.cerebra_volume[(self.brain_volume != 0) & (self.cerebra_volume == 0)] = 103
        # Add white matter to label details
        self.label_details.loc[len(self.label_details.index)] = [0, "White matter", 103]
        self.region_points_cache = {}

        # Metadata
        self.region_ids = np.sort(self.label_details["CerebrA ID"].unique())

        # TODO: Look into sparse representations
        # self.volume_data_sparse = {
        #     region_id: self.get_points_from_region_id(region_id)
        #     for region_id in self.region_ids
        # }

    def orthoview(self, **kwargs):
        fig, axs = orthoview(self.cerebra_volume, self.affine, **kwargs)

        if "pt" in kwargs.keys() and kwargs["pt"] is not None:
            reg_name = self.get_region_name_from_point(kwargs["pt"])
            reg_id = self.get_region_id_from_point(kwargs["pt"])
            fig.suptitle(f"{reg_name} id ({reg_id})")

        return axs

    def plot_region_orthoview(self, region_id, **kwargs):
        reg_name = self.get_region_name_from_region_id(region_id)
        reg_points = self.get_points_from_region_id(region_id)
        reg_centroid = self.find_region_centroid_from_name(reg_name)

        fig, axs = orthoview_region(
            reg_points,
            reg_centroid,
            self.cerebra_volume,
            self.affine,
            region_id,
            **kwargs,
        )

        fig.suptitle(reg_name)

        return axs

    def plot_3d(self):
        return plot_volume_3d(self.cerebra_volume, density=8)

    def plot_region_3d(self, region_id):
        pts = self.get_points_from_region_id(region_id)
        plot_volume_3d(self.cerebra_volume, region_pts=pts)

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
        if region_id not in self.region_points_cache.keys():
            self.region_points_cache[region_id] = np.array(
                np.where(self.cerebra_volume == region_id)
            ).T
        return self.region_points_cache[region_id]

    def get_region_name_from_region_id(self, region_id):
        if region_id == 0:
            return ""
        if region_id > 103:
            return "Non-valid id"
        return self.label_details[self.label_details["CerebrA ID"] == region_id][
            "Label Name"
        ].item()

    def get_closest_region_to_whitematter(self, pt, n_max=10):
        success = False

        region_id = self.get_region_id_from_point(pt)
        if region_id != 103:
            logging.warning(
                "Attempting to get closest region to whitematter from a non-whitematter region"
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

        return success, None, None

    def voxel_to_ras(self, pt):
        return mne.transforms.apply_trans(self.affine, pt).astype(int)

    def ras_to_voxel(self, pt):
        return np.round(
            mne.transforms.apply_trans(np.linalg.inv(self.affine), pt)
        )  # NOTE: removed .astype(int)

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
