#!/usr/bin/env python
"""
ForwardMNE submodule for cerebra_atlas_python
"""

from .mne_src_space import SourceSpaceMNE
from .mne_bem import BEMMNE
from .mne_montage import MontageMNE


class ForwardMNE(SourceSpaceMNE, BEMMNE, MontageMNE):
    def __init__(self, **kwargs):
        SourceSpaceMNE.__init__(self, **kwargs)
        BEMMNE.__init__(self, **kwargs)
        MontageMNE.__init__(self, **kwargs)
