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


def rgb_to_hex_str(color_rgb: np.ndarray) -> str:
    """Transforms (r,g,b) (0,1) array into hex color string

    Args:
        color_rgb (np.ndarray): input array

    Returns:
        str: transformed hex string
    """
    color_rgb_list = [int(c * 255) for c in color_rgb]
    return f"#{color_rgb_list[0]:02x}{color_rgb_list[1]:02x}{color_rgb_list[2]:02x}"


def get_cmap_colors(cmap_name="gist_rainbow", n_classes=103):
    n_colors = int(n_classes) + 1
    cmap = matplotlib.colormaps[cmap_name]
    colors = cmap(np.linspace(0, 1, n_colors))
    white = np.array([1, 0.87, 0.87, 1])
    colors[-1] = white
    black = np.array([0, 0, 0, 1])
    colors[0] = black
    return colors[:, :3]


def get_cmap_colors_hex(**kwargs):
    colors = get_cmap_colors()
    return np.array([rgb_to_hex_str(c) for c in colors])


def get_cmap():
    newcmp = ListedColormap(get_cmap_colors())
    return newcmp


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
