#!/usr/bin/env python
import unittest
import logging
import numpy as np
from cerebra_atlas_python.plotting.plotting import Plotting
from tests.test_base import TestBase


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
        cerebra_plotting = Plotting()
        logger.info("Plotting init ok")


if __name__ == "__main__":
    unittest.main()
