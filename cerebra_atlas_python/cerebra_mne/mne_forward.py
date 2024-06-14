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
    def __init__(self, cache_path:str, montage_name:str, head_size:float, fixed_ori:bool = False, meg:bool=False, eeg:bool = False, n_jobs:int = -1, cache_result:bool=True, **kwargs):
        SourceSpaceMNE.__init__(self, **kwargs)
        BEMMNE.__init__(self, **kwargs)

        self.fixed_ori = fixed_ori
        self.meg: bool = meg
        self.eeg: bool = eeg
        self.n_jobs: bool = n_jobs
        self.cache_result:bool = cache_result

        # Avoid recomputing/reloading fwd solution from disk
        self._forward:mne.Forward = None
        self._forward_path:str = op.join(cache_path, f"{montage_name}-hs{head_size}-fwd.fif")

    
