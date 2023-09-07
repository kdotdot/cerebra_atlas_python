import os
import logging
import nibabel as nib
import numpy as np
import pandas as pd
from cerebra_atlas_python.utils import download_file_from_google_drive, setup_logging

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
        cerebra_in_head_path = f"{download_dir}/CerebrA_in_head.mgz"
        label_details_path = f"{download_dir}/CerebrA_LabelDetails.csv"

        # Create download folder if it does not exist
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download data if it is not present within download folder
        if download_data and not os.path.exists(cerebra_in_head_path):
            logging.info("Downloading CerebrA volume...")
            file_id = "13rfrvxVQe18ss2hccPy10DkKQdnNyjWL"
            download_file_from_google_drive(file_id, cerebra_in_head_path)
        if download_data and not os.path.exists(label_details_path):
            logging.info("Downloading CerebrA labels...")
            file_id = "1RoOfEiqglZ6wM2gU6Qae48uc3j8DXv5d"
            download_file_from_google_drive(file_id, label_details_path)

        # Read data
        self.cerebra_img = nib.load(cerebra_in_head_path)
        self.volume_data = np.array(self.cerebra_img.dataobj)
        self.label_details = preprocess_label_details(pd.read_csv(label_details_path))

    # TODO: Specify which coordinate frame are xyz
    # Given a point in a (256,256,256) 3d space, determines
    # which brain region it belongs to
    def get_region_name_from_point(self, x, y, z):
        label_id = self.volume_data[x, y, z]
        print(label_id)
        if label_id == 0:
            return self.label_details[self.label_details["CerebrA ID"] == label_id]

        return

    # Given a brain region name, get an array of points that
    # make up said region
    def get_points_from_region_name(region_name):
        pass

    # Helper function for get_points_from_region_name
    # Does the same but with region id instead of region name
    def get_points_from_region_id(id):
        pass

    def get_brain():
        pass

    def plot_mri_point():
        pass


if __name__ == "__main__":
    setup_logging()
    cerebra = CerebrA()

    print(cerebra.volume_data.min(), cerebra.volume_data.max())
