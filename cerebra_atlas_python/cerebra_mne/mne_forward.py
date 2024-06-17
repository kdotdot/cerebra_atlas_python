#!/usr/bin/env python
"""
ForwardMNE submodule for cerebra_atlas_python
"""
import os.path as op
from .mne_src_space import SourceSpaceMNE
from .mne_bem import BEMMNE
from ..data import CerebraData


class ForwardMNE(SourceSpaceMNE, BEMMNE):
    def __init__(
        self,
        cache_path: str,
        cerebra_data: CerebraData,
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
        SourceSpaceMNE.__init__(
            self, cerebra_data=cerebra_data, cache_path=self.cache_path, **kwargs
        )
        BEMMNE.__init__(
            self,
            cache_path=self.cache_path,
            subjects_dir=cerebra_data.subjects_dir,
            **kwargs,
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
