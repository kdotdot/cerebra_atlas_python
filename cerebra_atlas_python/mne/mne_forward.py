#!/usr/bin/env python
"""
ForwardMNE submodule for cerebra_atlas_python
"""

from .mne_src_space import SourceSpaceMNE
from .mne_bem import BEMMNE
from .mne_montage import AlignedMontageMNE


class ForwardMNE(SourceSpaceMNE, BEMMNE, AlignedMontageMNE):
    def __init__(self, **kwargs):
        SourceSpaceMNE.__init__(self, **kwargs)
        BEMMNE.__init__(self, **kwargs)
        AlignedMontageMNE.__init__(self, **kwargs)
