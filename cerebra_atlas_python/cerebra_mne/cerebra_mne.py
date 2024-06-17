#!/usr/bin/env python
"""
MNE submodule for cerebra_atlas_python
Wraps source space, BEM, alignment, montage and Forward model
"""
import logging
import os.path as op
import tempfile
import mne
import os
from cerebra_atlas_python.data import CerebraData
from .mne_forward import ForwardMNE
from .mne_montage import MontageMNE

logger = logging.getLogger(__name__)


class MNE(ForwardMNE):
    def __init__(self, cerebra_data: CerebraData, **kwargs):
        self.cerebra_data = cerebra_data
        ForwardMNE.__init__(self, cerebra_data=self.cerebra_data, **kwargs)

    def get_forward(self, montage_name, head_size, sfreq=None):
        # Set info
        self.info = MontageMNE.get_info(
            montage_name=self.montage_name, head_size=self.head_size
        )
        # Set trans
        # Access forward
        return self.forward

    def _corregistration(self, montage_name, head_size, sfreq=None):
        info = MontageMNE.get_info(
            montage_name=montage_name, head_size=head_size, sfreq=None
        )
        info_path = op.join(tempfile.gettempdir(), "temp_info.fif")
        info.save(info_path)
        trans_default_path = op.join(
            self.cerebra_data.subjects_dir, self.cerebra_data.subject_name, "trans.fif"
        )
        logger.info(f"Save trans file to {trans_default_path} after aligment")
        trans_path = op.join(
            self.cerebra_data.subjects_dir,
            self.cerebra_data.subject_name,
            f"corregistration/{montage_name}_{head_size}_trans.fif",
        )
        logger.info(f"Will be automatically renamed to {trans_path}")
        mne.gui.coregistration(
            inst=info_path,
            subjects_dir=self.cerebra_data.subjects_dir,
            block=True,
            subject=self.cerebra_data.subject_name,
        )
        os.remove(info_path)
        if op.exists(trans_default_path):
            os.rename(trans_default_path, trans_path)
