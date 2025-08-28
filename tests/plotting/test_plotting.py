#!/usr/bin/env python
import unittest
import logging
import numpy as np
from tests.test_base import TestBase
from cerebra_atlas_python.plotting.colors import normalize_colors_input

logger = logging.getLogger(__name__)


class TestData(TestBase):
    """
    Test methods in
    cerebra_atlas_python/plotting
    """

    def test_plotting(self):
        """
        Test methods
        """
        logger.info("Plotting init ok")

    def test_colors_input(self):
        self.assertIsNot(normalize_colors_input("#ffffff", 10), None)
        self.assertIs(len(normalize_colors_input("#ffffff", 10)), 10)


if __name__ == "__main__":
    unittest.main()
