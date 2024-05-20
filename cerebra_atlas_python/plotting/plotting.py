#!/usr/bin/env python
"""
Plotting submodule for cerebra_atlas_python
Wraps 2D and 3D plotting
"""

from .plotting_2d import Plots2D
from .plotting_3d import Plots3D


class Plotting(Plots2D, Plots3D):

    bem_colors = [[0, 0.1, 1], [0.1, 0.2, 0.9], [0.2, 0.1, 0.95]]
    cortical_color = [0.3, 1, 0.5]
    non_cortical_color = [1, 0.4, 0.3]

    def __init__(self, **kwargs):
        Plots2D.__init__(
            self,
            **kwargs,
        )
