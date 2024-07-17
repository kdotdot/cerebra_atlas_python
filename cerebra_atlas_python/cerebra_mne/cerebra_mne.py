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

import numpy as np
from cerebra_atlas_python.data import CerebraData
from .mne_forward import ForwardMNE
from .mne_montage import MontageMNE
from ..data._transforms import apply_trans

logger = logging.getLogger(__name__)


class MNE(ForwardMNE):
    def __init__(
        self, cerebra_data: CerebraData, montage_name=None, head_size=None, **kwargs
    ):
        self.cerebra_data = cerebra_data
        ForwardMNE.__init__(self, cerebra_data=self.cerebra_data, **kwargs)

        self.montage_name = montage_name
        self.head_size = head_size
        self.sfreq = None

    @property
    def trans_path(self):
        return op.join(
            self.cerebra_data.subjects_dir,
            self.cerebra_data.subject_name,
            f"corregistration/{self.montage_name}_{self.head_size}_trans.fif",
        )

    @property
    def info(self):
        assert (
            self.montage_name is not None and self.head_size is not None
        ), "Montage name and head size should be provided for montage info"
        return MontageMNE.get_info(
            montage_name=self.montage_name, head_size=self.head_size, sfreq=self.sfreq
        )

    @property
    def head_mri_trans(self) -> mne.Transform:
        # Set trans
        assert op.exists(
            self.trans_path
        ), f"self.trans_path does not exist:{self.trans_path}"
        self.trans = mne.read_trans(op.join(self.trans_path))
        return self.trans

    def apply_head_mri_trans(self, points):
        return apply_trans(self.head_mri_trans, points)

    def get_forward(self):

        # Access forward
        return self.forward

    def _corregistration(self, montage_name=None, head_size=None, sfreq=None):
        assert (
            montage_name is not None or self.montage_name is not None
        ), "Montage name should be provided for corregistration"
        montage_name = montage_name or self.montage_name
        assert (
            head_size is not None or self.head_size is not None
        ), "Head size should be provided for corregistration"
        head_size = head_size or self.head_size
        info = MontageMNE.get_info(
            montage_name=montage_name, head_size=head_size, sfreq=None
        )
        info_path = op.join(tempfile.gettempdir(), "temp_info.fif")
        info.save(info_path)
        trans_default_path = op.join(
            self.cerebra_data.subjects_dir, self.cerebra_data.subject_name, "trans.fif"
        )
        logger.info(f"Save trans file to {trans_default_path} after aligment")

        logger.info(f"Will be automatically renamed to {self.trans_path}")
        mne.gui.coregistration(
            inst=info_path,
            subjects_dir=self.cerebra_data.subjects_dir,
            block=True,
            subject=self.cerebra_data.subject_name,
        )
        os.remove(info_path)
        if op.exists(trans_default_path):
            os.rename(trans_default_path, self.trans_path)
