"""Main cerebra class
"""

import os.path as op
import appdirs
from .data import CerebraData
from .plotting import Plotting
from .cerebra_mne import MNE


class CerebrA(CerebraData, Plotting, MNE):
    """Main cerebra class SA"""

    def __init__(self, **kwargs):

        self.cache_path = op.join(
            appdirs.user_cache_dir("cerebra_atlas_python"), "cerebra"
        )

        CerebraData.__init__(self, cache_path=self.cache_path, **kwargs)
        Plotting.__init__(self, **kwargs)
        # SourceSpaceData should be initialized first
        MNE.__init__(self, cache_path=self.cache_path, cerebra_data=self, **kwargs)

    def corregistration(self, **kwargs):
        """Manually generate fiducials.fif and head-mri-trans.fif
        Saved to standard location (cerebra_data/FreeSurfer/bem/)
        """
        assert (
            "montage" in kwargs or "montage_name" in kwargs
        ), "Either MME montage or montage_name should be provided for corregistration"
        self._corregistration(**kwargs)

    def _prepare_plot_data(self, **kwargs):
        plot_data = {"affine": self.affine, "cerebra_volume": self.cerebra_volume}
        return plot_data

    def _plot(self, **kwargs):
        plot_data = self._prepare_plot_data(**kwargs)
        self._plot_data(plot_data=plot_data, **kwargs)

    def orthoview(self, **kwargs):
        """Plot 2D brain with orthoview"""
        self._plot(kind="orthoview", **kwargs)

    def plot2d(self, **kwargs):
        """Plot 2D brain"""
        self._plot(kind="2d", **kwargs)

    def plot3d(self, **kwargs):
        """Plot 3D brain"""
        self._plot(kind="3d", **kwargs)


if __name__ == "__main__":
    cerebra = CerebrA()
    cerebra.corregistration()
