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
            cache_path=self.cerebra.cache_path, cerebra_data=self.cerebra.cerebra_data
        )
        logger.info("MNE init ok")

    def test_bem(self):
        """
        Test methods
        """
        cerebra_mne = MNE(
            cache_path=self.cerebra.cache_path, cerebra_data=self.cerebra.cerebra_data
        )
        logger.info("MNE init ok")
        self.assertIsNot(cerebra_mne.bem, None)
        self.assertIsNot(cerebra_mne.bem_model, None)
        logger.info("BEM ok")
        self.assertEqual(
            cerebra_mne.get_bem_vertices_mri().shape,
            cerebra_mne.get_bem_normals_mri().shape,
        )
        self.assertIsNot(cerebra_mne.get_bem_vertices_mri(), None)
        self.assertIsNot(cerebra_mne.get_bem_triangles(), None)


if __name__ == "__main__":
    unittest.main()
