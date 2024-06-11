#!/usr/bin/env python
"""
MNE submodule for cerebra_atlas_python
Wraps source space, BEM, alignment, montage and Forward model
"""
import os.path as op
import tempfile
import mne
import os
from .mne_forward import ForwardMNE


class MNE(ForwardMNE):
    def __init__(self, **kwargs):
        ForwardMNE.__init__(self, **kwargs)

    def _corregistration(self, subjects_dir, **kwargs):
        info = self.get_info(**kwargs)
        info_path = op.join(tempfile.gettempdir(), "temp_info.fif")
        info.save(info_path)
        mne.gui.coregistration(inst=info_path, subjects_dir=subjects_dir, block=True)
        os.remove(info_path)
