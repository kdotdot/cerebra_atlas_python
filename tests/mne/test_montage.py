#!/usr/bin/env python
import unittest
import logging
import numpy as np
from cerebra_atlas_python.cerebra_mne.mne_montage import MontageMNE
from tests.test_base import TestBase


logger = logging.getLogger(__name__)


class TestData(TestBase):
    """
    Test methods in
    cerebra_atlas_python/plotting
    """

    def test_plotting(self):
        """
        Test montages
        """
        # Pylint allow protected access
        # pylint: disable=W0212
        montage1 = MontageMNE._get_standard_montage(
            kept_ch_names=None, kind="GSN-HydroCel-129", head_size=0.108
        )
        montage2 = MontageMNE._get_standard_montage(
            kept_ch_names=None, kind="GSN-HydroCel-129", head_size=0.101
        )
        montage3 = MontageMNE._get_standard_montage(
            kept_ch_names=None, kind="standard_1005", head_size=0.108
        )
        logging.info(montage1.dig[:3][0]["r"])
        logging.info(montage2.dig[:3][0]["r"])
        # Make sure head size works
        self.assertTrue(
            (montage1.dig[:3][0]["r"] != montage2.dig[:3][0]["r"]).sum() == 3
        )
        # Make sure different montages are different
        self.assertTrue(montage1.ch_names != montage3.ch_names)
        self.assertTrue(montage1.ch_names == montage2.ch_names)

        montage4 = MontageMNE.get_montage(
            montage_name="GSN-HydroCel-129-downsample-111", head_size=0.108
        )
        self.assertTrue(
            (montage4.dig[:3][0]["r"] == montage4.dig[:3][0]["r"]).sum() == 3
        )
        montage5 = MontageMNE.get_montage(
            montage_name="GSN-HydroCel-129", head_size=0.108
        )
        self.assertTrue(montage1.ch_names == montage5.ch_names)

        info = MontageMNE.get_info()
        self.assertIsNot(info, None)


if __name__ == "__main__":
    unittest.main()
