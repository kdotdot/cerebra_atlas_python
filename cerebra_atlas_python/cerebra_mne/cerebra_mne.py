#!/usr/bin/env python
"""
MNE submodule for cerebra_atlas_python
Wraps source space, BEM, alignment, montage and Forward model
"""
import os
import os.path as op
import tempfile
import mne
from .mne_forward import ForwardMNE
from .mne_montage import MontageMNE


class MNE(ForwardMNE):
    def __init__(self,subjects_dir:str, **kwargs):
        self.subjects_dir = subjects_dir
        ForwardMNE.__init__(self, **kwargs)

    def _corregistration(self, **kwargs):
        info = MontageMNE.get_info(**kwargs)
        info_path = op.join(tempfile.gettempdir(), "temp_info.fif")
        info.save(info_path)
        mne.gui.coregistration(inst=info_path, subjects_dir=self.subjects_dir, block=True)
        os.remove(info_path)
