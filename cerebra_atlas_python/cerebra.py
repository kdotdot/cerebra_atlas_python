import os
import logging
import nibabel as nib
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from cerebra_atlas_python.utils import download_file_from_google_drive, setup_logging
from cerebra_atlas_python.plotting import imshow_mri

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
        brain_path = "/home/carlos/Datasets/subjects/MNIAverage/mri/brain.mgz"

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
        self.cerebra_img = nib.load(cerebra_in_head_path)  # LIA coordinate frame
        self.volume_data = np.array(self.cerebra_img.dataobj)

        self.label_details = preprocess_label_details(pd.read_csv(label_details_path))

        self.brain_img = nib.load(brain_path)  # LIA coordinate frame
        self.brain_data = np.array(self.brain_img.dataobj)

        self.t1_img = nib.load(t1_path)  # LIA coordinate frame
        self.t1_data = np.array(self.t1_img.dataobj)

        # Add whitematter to volume data
        self.volume_data[(self.brain_data != 0) & (self.volume_data == 0)] = 103
        # Add white matter to label details
        self.label_details.loc[len(self.label_details.index)] = [
            None,
            "White Matter",
            103,
        ]

        # Move to RAS coordinate frame
        self.volume_data = np.rot90(self.volume_data, -1, axes=(1, 2))
        self.volume_data = np.flipud(self.volume_data)

        self.region_ids = np.sort(self.label_details["CerebrA ID"].unique())

        # TODO: Look into sparse representations
        # self.volume_data_sparse = {
        #     region_id: self.get_points_from_region_id(region_id)
        #     for region_id in self.region_ids
        # }

    # Given a point in a (256,256,256) 3d space, determines
    # which brain region it belongs to
    # Point
    def get_region_name_from_point(self, point):
        label_id = self.volume_data[x, y, z]
        if label_id == 0:
            return self.label_details[self.label_details["CerebrA ID"] == label_id]

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
        return np.array(np.where(self.volume_data == region_id)).T

    def get_region_name_from_region_id(self, region_id):
        return self.label_details[self.label_details["CerebrA ID"] == region_id][
            "Label Name"
        ].item()

    def get_closest_region_to_whitematter(self, x, y, z):
        pass

    def voxel_to_ras(self, x, y, z):
        return mne.transforms.apply_trans(self.cerebra_img.affine, np.array([x, y, z]))

    def ras_to_voxel(self, x, y, z):
        return np.round(
            mne.transforms.apply_trans(
                np.linalg.inv(self.cerebra_img.affine), np.array([x, y, z])
            )
        ).astype(int)

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

    def get_region_slices_from_id(self, region_id):
        centroid = self.find_region_centroid_from_id(region_id)
        points = self.get_points_from_region_id(region_id)
        mask1 = points.T[0] == centroid[0]
        mask2 = points.T[1] == centroid[1]
        mask3 = points.T[2] == centroid[2]
        slice1 = points[mask1]
        slice2 = points[mask2]
        slice3 = points[mask3]
        return [slice1, slice2, slice3], centroid

    def plot_mri_vox(self, x, y, z, title="MRI slice", plot_all_ax=True, slices=None):
        vox = self.voxel_to_ras(x, y, z)
        return self.plot_mri_ras(vox[0], vox[1], vox[2], title, plot_all_ax, slices)

    def get_slices(self, ijk):
        i, j, k = ijk

        mask1 = self.volume_data == centroid[0]
        mask2 = volume_data == centroid[1]
        mask3 = points.T[2] == centroid[2]
        slice1 = points[mask1]
        slice2 = points[mask2]
        slice3 = points[mask3]

    def orthoview(self, i=128, j=128, k=128):
        slices = self.get_slices([i, j, k])

        pass

    def plot_mri_ras(self, x, y, z, title="MRI slice", plot_all_ax=True, slices=None):
        if not plot_all_ax:
            fig = imshow_mri(
                self.t1_data,
                self.t1_img,
                self.ras_to_voxel(x, y, z),
                {"Scanner RAS": np.array([x, y, z])},
                title,
                orientation_axis=0,
                slices=slices,
            )
        else:
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            axs[-1, -1].xaxis.set_visible(False)
            axs[-1, -1].yaxis.set_visible(False)
            axs[-1, -1].spines["top"].set_visible(False)
            axs[-1, -1].spines["right"].set_visible(False)
            axs[-1, -1].spines["bottom"].set_visible(False)
            axs[-1, -1].spines["left"].set_visible(False)
            for i in range(3):
                imshow_mri(
                    self.t1_data,
                    self.t1_img,
                    self.ras_to_voxel(x, y, z),
                    {"Scanner RAS": np.array([x, y, z])},
                    title,
                    orientation_axis=i,
                    ax=axs[i if i < 2 else 0, 1 if i == 2 else 0],
                    slices=slices,
                ),

            fig.suptitle(title)
            fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)
        return fig


if __name__ == "__main__":
    setup_logging()
    cerebra = CerebrA()

    print(cerebra.volume_data.min(), cerebra.volume_data.max())
