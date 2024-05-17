#!/usr/bin/env python
"""
Plotting submodule for cerebra_atlas_python
Wraps 2D and 3D plotting
"""

from .plotting_2d import Plots2D
from .plotting_3d import Plots3D

class Plotting(Plots2D, Plots3D):
    def __init__(self, **kwargs):
        Plots2D.__init__(self,**kwargs,
        )
        