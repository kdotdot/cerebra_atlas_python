#!/usr/bin/env python
import unittest
import logging
import numpy as np
from ..test_base import TestBase

from cerebra_atlas_python.plotting.plotting import Plotting

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

   


if __name__ == "__main__":
    unittest.main()
