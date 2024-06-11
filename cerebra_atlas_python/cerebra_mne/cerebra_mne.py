#!/usr/bin/env python
"""
MNE submodule for cerebra_atlas_python
Wraps source space, BEM, alignment, montage and Forward model
"""

from .mne_forward import ForwardMNE


class MNE(ForwardMNE):
    def __init__(self, **kwargs):
        ForwardMNE.__init__(self, **kwargs)
