"""Main cerebra class
"""

import mne
from cerebra_atlas_python.data import SourceSpaceData
from cerebra_atlas_python.plotting import Plotting
from cerebra_atlas_python.cerebra_mne import MNE


class CerebrA(SourceSpaceData, Plotting, MNE):
    """Main cerebra class SA"""

    def __init__(self, **kwargs):
        SourceSpaceData.__init__(self, **kwargs)
        Plotting.__init__(self, **kwargs)
        MNE.__init__(self, **kwargs)

    def generate_fiducials(self, **kwargs):
        """Generate fiducials"""
        mne.gui.coregistration(block=True)


if __name__ == "__main__":
    cerebra = CerebrA()
    cerebra.generate_fiducials()
