"""Main cerebra class
"""

from .data import SourceSpaceData
from .plotting import Plotting
from .mne import MNE


class CerebrA(SourceSpaceData, Plotting, MNE):
    """Main cerebra class SA"""

    def __init__(self, **kwargs):
        SourceSpaceData.__init__(self, **kwargs)
        Plotting.__init__(self, **kwargs)
        MNE.__init__(self, **kwargs)
