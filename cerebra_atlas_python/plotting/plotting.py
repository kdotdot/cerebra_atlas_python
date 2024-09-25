#!/usr/bin/env python
"""
Plotting submodule for cerebra_atlas_python
Wraps 2D and 3D plotting
"""

import matplotlib
from matplotlib.colors import ListedColormap
import numpy as np

from cerebra_atlas_python.plotting.colors import get_cmap_colors_hex
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
        Plots3D.__init__(
            self,
            **kwargs,
        )

    def _add_colors_to_plot_data(self, plot_data):
        plot_data = {
            **plot_data,
            "bem_colors": self.bem_colors,
            "cortical_color": self.cortical_color,
            "non_cortical_color": self.non_cortical_color,
        }
        return plot_data

    def _plot_data(self, kind: str, plot_data, **kwargs):
        """Plot data based on kind

        Args:
            kind (str): "orthoview", "2d", or "3d"

        Raises:
            ValueError: Unsupported kind
        """
        plot_data = self._add_colors_to_plot_data(plot_data)
        if kind == "2d":
            self.plot_data_2d(plot_data=plot_data, **kwargs)
        elif kind == "orthoview":
            self.plot_data_orthoview(plot_data=plot_data, **kwargs)
        elif kind == "3d":
            self.plot_data_3d(plot_data=plot_data, **kwargs)
        else:
            raise ValueError(f"kind {kind} not supported")

    def get_cortical_colors(self, rgba=False, rgb=False):
        if rgba and rgb:
            raise ValueError("Only one of {rgba,rgb} should be True")
        colors = get_cmap_colors_hex()
        if rgb:
            colors = [matplotlib.colors.to_rgb(c) for c in colors]
        if rgba:
            colors = [matplotlib.colors.to_rgba(c) for c in colors]
        cortical_colors = [
            colors[region_id] for region_id in self.get_cortical_region_ids()
        ]
        return cortical_colors
