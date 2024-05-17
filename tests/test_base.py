#!/usr/bin/env python
import unittest
import pathlib as pl
from cerebra_atlas_python import setup_logging


class TestBase(unittest.TestCase):
    """
    Base class for test methods
    """

    def __init__(self, *args, **kwargs):
        """Init for all tests"""
        setup_logging("DEBUG")
        super(TestBase, self).__init__(*args, **kwargs)

    def assertIsFile(self, path):
        """Assert file exists"""
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))


if __name__ == "__main__":
    unittest.main()
