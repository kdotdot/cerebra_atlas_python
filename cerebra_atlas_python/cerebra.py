"""Main cerebra class
"""

import os.path as op
import appdirs
from .data import SourceSpaceData
from .plotting import Plotting
from .cerebra_mne import MNE


class CerebrA(SourceSpaceData, Plotting, MNE):
    """Main cerebra class SA"""

    def __init__(self, **kwargs):

        self.cache_path = op.join(
            appdirs.user_cache_dir("cerebra_atlas_python"), "cerebra"
        )

        SourceSpaceData.__init__(self, **kwargs)

        Plotting.__init__(self, **kwargs)
        MNE.__init__(self, **kwargs)

    def corregistration(self, **kwargs):
        """Manually generate fiducials.fif and head-mri-trans.fif
        Saved to standard location (cerebra_data/FreeSurfer/bem/)
        """
        assert (
            "montage" in kwargs or "montage_name" in kwargs
        ), "Either MME montage or montage_name should be provided for corregistration"
        self._corregistration(subjects_dir=self.subjects_dir, **kwargs)


if __name__ == "__main__":
    cerebra = CerebrA()
    cerebra.generate_fiducials()
