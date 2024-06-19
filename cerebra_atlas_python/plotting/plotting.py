#!/usr/bin/env python
"""
Plotting submodule for cerebra_atlas_python
Wraps 2D and 3D plotting
"""

import matplotlib
from matplotlib.colors import ListedColormap
import numpy as np
from .plotting_2d import Plots2D
from .plotting_3d import Plots3D

# plot_kind = ["2d", "3d", "orthoview"]


class Plotting(Plots2D, Plots3D):

    bem_colors = [[0, 0.1, 1], [0.1, 0.2, 0.9], [0.2, 0.1, 0.95]]
    cortical_color = [0.3, 1, 0.5]
    non_cortical_color = [1, 0.4, 0.3]

    def __init__(self, **kwargs):
        Plots2D.__init__(
            self,
            **kwargs,
        )

    def _plot_data(self, kind: str, **kwargs):
        """Plot data based on kind

        Args:
            kind (str): "orthoview", "2d", or "3d"

        Raises:
            ValueError: Unsupported kind
        """
        if kind == "2d":
            self.plot_data_2d(**kwargs)
        elif kind == "orthoview":
            self.plot_data_orthoview(**kwargs)
        elif kind == "3d":
            self.plot_data_3d(**kwargs)
        else:
            raise ValueError(f"kind {kind} not supported")
