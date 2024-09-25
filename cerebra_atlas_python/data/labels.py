#!/usr/bin/env python
"""
This module handles stuff related with data/cerebra_data/label_details.csv
"""

import os.path as op
import pandas as pd
import numpy as np


class Labels:
    """Contains region information for each label within the atlas.
    Label 0 represents empty, label 103 represents whitematter
    Labels 1-102 are cerebral regions. 1-61 are right hemisphere
    and 62-102 are left hemisphere.

    Attributes:
        region_ids (np.ndarray): Region ids np.arange([0,104])
    """

    def __init__(self, cerebra_data_path: str, csv_name="label_details.csv", **kwargs):
        """_summary_

        Args:
            cerebra_data_path (str): _description_
        """
        self._label_details_path = op.join(cerebra_data_path, csv_name)
        self._label_details = pd.read_csv(self._label_details_path, index_col=0)
        # Metadata
        self.region_ids = np.sort(self._label_details["CerebrA ID"].unique())

    def _is_valid_region_id(self, region_id: int) -> bool:
        """Checks if region id is valid

        Args:
            region_id (int): region id

        Raises:
            ValueError: If region_id is not within the range of 0-103

        Returns:
            bool: is valid region id
        """
        if region_id < 0 or region_id > 103:
            raise ValueError(f"Region ID {region_id} not within range 0-103")
        return region_id in self.region_ids

    def get_region_data_from_region_id(self, region_id: int) -> pd.DataFrame:
        """Returns region data row matching region id"""
        assert self._is_valid_region_id(region_id)
        region_data = self._label_details[
            self._label_details["CerebrA ID"] == region_id
        ]

        if len(region_data) < 1:
            raise ValueError(f"Region ID {region_id} not found in label_details.csv")
        return region_data

    def get_region_data_from_region_name(self, region_name: str) -> pd.DataFrame:
        """Returns region data row matching region name"""
        region_data = self._label_details[
            self._label_details["Label Name"] == region_name
        ]
        return region_data

    def get_region_id_from_region_name(self, region_name: str) -> int:
        """Returns region data id matching region name"""
        region_data = self._label_details[
            self._label_details["Label Name"] == region_name
        ]
        return region_data["CerebrA ID"].item()

    def get_region_name_from_region_id(self, region_id: int) -> str:
        """Returns region name matching region id"""
        if region_id == 0:
            return "Empty region"
        if region_id > 103:
            return "Non-valid id"
        return self._label_details[self._label_details["CerebrA ID"] == region_id][
            "Label Name"
        ].item()

    def get_cortical_region_ids(self, hemisphere=None) -> np.ndarray:
        """Return cortical region ids as a numpy array

        Args:
            hemisphere ("str", optional): Can be "left", "right" or None to use both. Defaults to None.

        Raises:
            ValueError: Raises value error if hemisphere is not "left", "right" or None

        Returns:
            np.ndarray: cortical region ids
        """
        mask = self._label_details["cortical"].astype(bool)
        # This should be .fillna instead of .astype
        # FutureWarning: Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version.
        mask[0] = False  # Empty region
        mask[103] = False  # White matter

        if hemisphere is not None:
            if hemisphere == "left":
                mask = mask & (self._label_details["hemisphere"] == "Left")
            elif hemisphere == "right":
                mask = mask & (self._label_details["hemisphere"] == "Right")
            else:
                raise ValueError(f"Unknown hemisphere {hemisphere=}")

        cortical_labels = self._label_details[mask]["CerebrA ID"].to_numpy()
        return cortical_labels

    def get_non_cortical_region_ids(self, hemisphere=None) -> np.ndarray:
        """Return non cortical region ids as a numpy array

        Args:
            hemisphere ("str", optional): Can be "left", "right" or None to use both. Defaults to None.

        Raises:
            ValueError: Raises value error if hemisphere is not "left", "right" or None

        Returns:
            np.ndarray: non cortical region ids
        """

        cortical_region_ids = self.get_cortical_region_ids(hemisphere)
        non_cortical_region_ids = np.setdiff1d(self.region_ids, cortical_region_ids)
        return non_cortical_region_ids

    def get_cortical_id_from_region_id(self, region_id):
        return (np.where(self.get_cortical_region_ids() == region_id)[0][0]) + 1

    def get_region_id_from_cortical_region_id(self, cortical_region_id):
        return self.get_cortical_region_ids()[cortical_region_id - 1]
