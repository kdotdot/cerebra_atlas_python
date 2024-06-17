#!/usr/bin/env python
"""
ForwardMNE submodule for cerebra_atlas_python
"""
import os.path as op
import mne
from .mne_src_space import SourceSpaceMNE
from .mne_bem import BEMMNE
from .mne_montage import MontageMNE


class ForwardMNE(SourceSpaceMNE, BEMMNE):
    def __init__(
        self,
        cache_path: str,
        subjects_dir: str,
        montage_name: str | None = None,
        head_size: float | None = None,
        fixed_ori: bool = False,
        meg: bool = False,
        eeg: bool = False,
        n_jobs: int = -1,
        cache_result: bool = True,
        **kwargs,
    ):
        self.cache_path = cache_path
        self.subjects_dir = subjects_dir
        SourceSpaceMNE.__init__(self, **kwargs)
        BEMMNE.__init__(
            self, cache_path=self.cache_path, subjects_dir=self.subjects_dir, **kwargs
        )

        # self.fixed_ori = fixed_ori
        # self.meg: bool = meg
        # self.eeg: bool = eeg
        # self.n_jobs: int = n_jobs
        # self.cache_result: bool = cache_result

        # # Avoid recomputing/reloading fwd solution from disk
        # self._forward: mne.Forward | None = None
        self._forward_path: str = op.join(
            self.cache_path, f"{montage_name}-hs{head_size}-fwd.fif"
        )
