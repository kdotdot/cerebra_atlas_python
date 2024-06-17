#!/usr/bin/env python
import unittest
import logging
import numpy as np
from cerebra_atlas_python.cerebra_mne.cerebra_mne import MNE
from tests.test_base import TestBase
from cerebra_atlas_python import CerebrA


logger = logging.getLogger(__name__)


class TestData(TestBase):
    """
    Test methods in
    cerebra_atlas_python/plotting
    """

    def __init__(self, *args, **kwargs):
        super(TestData, self).__init__(*args, **kwargs)
        self.cerebra = CerebrA()

    def test_mne(self):
        """
        Test methods
        """
        cerebra_mne = MNE(
            cache_path=self.cerebra.cache_path, subjects_dir=self.cerebra.subjects_dir
        )
        logger.info("MNE init ok")


if __name__ == "__main__":
    unittest.main()
