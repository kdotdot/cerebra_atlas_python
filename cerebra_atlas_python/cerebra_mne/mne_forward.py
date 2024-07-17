#!/usr/bin/env python
"""
ForwardMNE submodule for cerebra_atlas_python
"""
from functools import cached_property
import os
from typing import Optional
import mne
import logging
import os.path as op
from .mne_src_space import SourceSpaceMNE
from .mne_bem import BEMMNE
from ..data import CerebraData
from ..data._cache import cache_mne_forward
from .mne_montage import MontageMNE

logger = logging.getLogger(__name__)


def get_forward_fsaverage(
    trans=None, src_space=None, bem=None, info=None, meg=False, eeg=True, n_jobs=-1
):
    if info is None:
        info = MontageMNE.get_info()
    if (trans is None) or (src_space is None) or (bem is None):
        fs_dir = mne.datasets.fetch_fsaverage()
        trans_fif_path = os.path.join(fs_dir, "bem", "fsaverage-trans.fif")
        src_fif_path = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
        bem_fif_path = os.path.join(
            fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif"
        )

        if trans is None:
            logging.warning("trans not provided, using fsaverage")
            trans = trans_fif_path
        if src_space is None:
            logging.warning("src_space not provided, using fsaverage")
            # src = mne.setup_source_space(subject, spacing=sampling, surface='white',
            #                         subjects_dir=subjects_dir, add_dist=False,
            #                         n_jobs=-1)
            src_space = src_fif_path
        if bem is None:
            logging.warning("bem not provided, using fsaverage")
            bem = bem_fif_path

    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src_space,
        bem=bem,
        meg=meg,
        eeg=eeg,
        n_jobs=n_jobs,
    )
    return fwd


def get_forward(
    trans, src_space, bem, info, meg=False, eeg=True, n_jobs=-1
) -> mne.Forward:

    print("==================EEG", eeg)
    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src_space,
        bem=bem,
        meg=meg,
        eeg=eeg,
        n_jobs=n_jobs,
    )
    assert fwd is not None, "Forward solution is None. Check inputs."

    return fwd


class ForwardMNE(SourceSpaceMNE, BEMMNE):
    def __init__(
        self,
        cache_path: str,
        cerebra_data: CerebraData,
        montage_name: Optional[str] = None,
        head_size: Optional[float] = None,
        fixed_ori: bool = False,
        meg: bool = False,
        eeg: bool = True,
        n_jobs: int = -1,
        cache_result: bool = True,
        **kwargs,
    ):
        self.cache_path = cache_path
        SourceSpaceMNE.__init__(
            self, cerebra_data=cerebra_data, cache_path=self.cache_path, **kwargs
        )
        BEMMNE.__init__(
            self,
            cache_path=self.cache_path,
            subjects_dir=cerebra_data.subjects_dir,
            **kwargs,
        )

        self.montage_name = montage_name
        self.head_size = head_size

        self.fixed_ori = fixed_ori
        self.meg: bool = meg
        self.eeg: bool = eeg
        self.n_jobs: int = n_jobs
        self.cache_result: bool = cache_result

        self.trans = None

        # # Avoid recomputing/reloading fwd solution from disk
        # self._forward: mne.Forward | None = None

    def assert_all_set(self):
        if self.trans is None:
            raise ValueError("trans is not set")
        if self.src_space is None:
            raise ValueError("src_space is not set")
        if self.bem is None:
            raise ValueError("bem is not set")
        if self.montage_name is None or self.head_size is None:
            raise ValueError("Info is not set. (montage_name or head_size) is not set")

    @property
    def fwd_string(self):
        return f"{self.montage_name}_{self.head_size}_{self.src_space_string}_{self.bem_string}-fwd"

    @property
    def forward(self):
        # Source ori 2 = "FIFFV_MNE_FREE_ORI"
        # if self._forward is not None and self._forward["source_ori"] == 2 and self.fixed_ori:
        def compute_fn(self):
            self.assert_all_set()
            logger.debug("Generating forward solution")
            # if self.info is None:
            #     self.info = MontageMNE.get_info(
            #         montage_name=self.montage_name, head_size=self.head_size
            #     )
            fwd = get_forward(
                src_space=self.src_space,
                bem=self.bem,
                info=self.info,
                meg=self.meg,
                eeg=self.eeg,
                n_jobs=self.n_jobs,
                trans=self.trans,
            )
            return fwd

        forward_path: str = op.join(self.cache_path, f"{self.fwd_string}.fif")
        print(forward_path)
        return cache_mne_forward(compute_fn, forward_path, self.fixed_ori, self)
