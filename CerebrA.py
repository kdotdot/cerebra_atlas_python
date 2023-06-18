import os
import logging
import nibabel as nib
import numpy as np
import pandas as pd
from utils import download_file_from_google_drive

# Download: 
# https://drive.google.com/file/d/13rfrvxVQe18ss2hccPy10DkKQdnNyjWL/view?usp=sharing
# https://drive.google.com/file/d/1RoOfEiqglZ6wM2gU6Qae48uc3j8DXv5d/view?usp=sharing

class CerebrA:
    def __init__(self, download_dir="./cerebra_data", download_data=True):

        # Define paths for required data
        cerebra_in_head_path = f"{download_dir}/CerebrA_in_head.mgz"
        label_details_path = f"{download_dir}/CerebrA_LabelDetails.csv"

        # Create download folder if it does not exist
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # Download data if it is not present within download folder
        if download_data and not os.path.exists(cerebra_in_head_path):
            logging.info("Downloading CerebrA volume...")
            file_id = '13rfrvxVQe18ss2hccPy10DkKQdnNyjWL'
            download_file_from_google_drive(file_id, cerebra_in_head_path)
        if download_data and not os.path.exists(label_details_path):
            logging.info("Downloading CerebrA labels...")
            file_id = '1RoOfEiqglZ6wM2gU6Qae48uc3j8DXv5d'
            download_file_from_google_drive(file_id, label_details_path)

        # Read data
        self.cerebra_img = nib.load(cerebra_in_head_path)
        self.volume_data = np.array(self.cerebra_img.dataobj)
        self.label_details = pd.read_csv(label_details_path)

def setup_logging(level=logging.DEBUG):
    logging.basicConfig(
        level=level,
        format=" [%(levelname)s] %(asctime)s.%(msecs)02d %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

if __name__ == "__main__":
    setup_logging()
    cerebra = CerebrA()

    print(cerebra.volume_data.min(), cerebra.volume_data.max())
