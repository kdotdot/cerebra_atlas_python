"""
CerebrA has three main functionalities:

1) Accessing and interacting with the
mni_icbm_2009c average brain volume and its associated CerebrA atlas
easily and fast (cerebra_atlas_python.data).

2) Generate a forward model (MNE) with different configuration
options. Allows defining new EEG montages (cerebra_atlas_python.cerebra_mne).

3) Provide plotting utilities for the brain volume as well as stats
and graphs for raw EEG/localized sources (cerebra_atlas_python.plotting).
"""

from .cerebra import CerebrA

# from cerebra_atlas_python.mni_average import MNIAverage
from ._logging import setup_logging


# from .mni_average import MNIAverage
