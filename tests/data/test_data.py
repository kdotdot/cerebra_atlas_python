#!/usr/bin/env python
import unittest
import logging
import numpy as np
from cerebra_atlas_python.data.freesurfer import FreeSurfer
from cerebra_atlas_python.data.labels import Labels
from cerebra_atlas_python.data.image import Image
from cerebra_atlas_python.data.cerebra_data import CerebraData
from cerebra_atlas_python.data.source_space import SourceSpaceData

from cerebra_atlas_python.data._cache import _add_fn_hash_to_path
from tests.test_base import TestBase

logger = logging.getLogger(__name__)


class TestData(TestBase):
    """
    Test methods in
    cerebra_atlas_python/data/labels.py
    cerebra_atlas_python/data/freesurfer.py
    """

    def test_cerebralabels(self):
        """
        Test methods
        """
        cerebra_data = CerebraData()
        cerebra_labels = Labels(cerebra_data.cerebra_data_path)
        for i in range(104):  # Assert 104 regions exist in df
            cerebra_labels.get_region_data_from_region_id(i)
            cerebra_labels.get_region_name_from_region_id(i)
        self.assertEqual(cerebra_labels.get_cortical_region_ids().shape[0], 62)
        self.assertEqual(cerebra_labels.get_non_cortical_region_ids().shape[0], 42)

    def test_cerebraimage(self):
        """
        Test methods
        """
        cerebra_data = CerebraData()
        cerebra_image = Image(cerebra_data.cerebra_data_path)
        self.assertNotEqual(cerebra_image.cerebra_img, None)
        vol, affine = cerebra_image.get_cerebra_vox_affine_ras()
        self.assertIsNot(vol, None)
        self.assertIsNot(affine, None)
        data = np.random.rand(3, 3, 3)
        self.assertIsNot(cerebra_image.apply_vox_ras_t(data), None)
        self.assertIsNot(cerebra_image.apply_ras_vox_t(data), None)
        self.assertIsNot(cerebra_image.apply_vox_mri_t(data), None)
        self.assertIsNot(cerebra_image.apply_mri_vox_t(data), None)
        self.assertIsNot(cerebra_image.apply_mri_ras_t(data), None)
        self.assertIsNot(cerebra_image.apply_ras_mri_t(data), None)

    def test_freesurfer(self):
        """
        Test methods
        """
        cerebra_data = CerebraData()
        fs = FreeSurfer(cerebra_data.cerebra_data_path)
        self.assertNotEqual(fs.t1_img, None)
        self.assertNotEqual(fs.wm_img, None)
        self.assertNotEqual(fs.fiducials, None)
        self.assertNotEqual(fs.head_mri_trans, None)

    def test_cache(self):
        """
        Test methods
        """
        cerebra_data = CerebraData()
        self.assertIsNot(cerebra_data.cerebra_volume, None)
        self.assertIsNot(cerebra_data.affine, None)
        self.assertIsNot(cerebra_data.cerebra_sparse, None)

        def test_fn():
            return 1

        path_1 = _add_fn_hash_to_path(test_fn, "test.pkl")
        path_2 = _add_fn_hash_to_path(test_fn, "test.pkl")
        self.assertEqual(path_1, path_2)

        def test_fn_2(a):
            return 2

        path_3 = _add_fn_hash_to_path(test_fn_2, "test.pkl")
        self.assertNotEqual(path_1, path_3)

        self.assertIsFile(
            _add_fn_hash_to_path(
                cerebra_data._get_wm_cerebra_volume_ras,
                cerebra_data._cerebra_volume_path,
            )
        )

    def test_src_space(self):
        """
        Test methods
        """
        src_space = SourceSpaceData()
        self.assertIsNot(src_space.src_space_points, None)
        self.assertIsNot(src_space.src_space_labels, None)
        self.assertIsNot(src_space.src_space_points_lia, None)


if __name__ == "__main__":
    unittest.main()
