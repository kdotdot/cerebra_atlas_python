#!/usr/bin/env python
import unittest
import logging
from .test_base import TestBase

from cerebra_atlas_python import CerebrA

logger = logging.getLogger(__name__)


class TestData(TestBase):
    """
    Test methods
    """

    def test_cerebra(self):
        """
        Test methods
        """
        cerebra = CerebrA()
        self.assertIsNot(cerebra.cerebra_volume, None)


if __name__ == "__main__":
    unittest.main()