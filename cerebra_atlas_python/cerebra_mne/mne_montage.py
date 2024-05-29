#!/usr/bin/env python
"""
MontageMNE submodule for cerebra_atlas_python
"""
import mne
from .downsampled_montages import montage_names, electrode_names


class MontageMNE:
    montage_names = montage_names
    electrode_names = electrode_names

    @classmethod
    def _get_standard_montage(
        cls, kept_ch_names=None, kind="GSN-HydroCel-129", head_size=0.108
    ) -> mne.channels.DigMontage:
        """Get a standard MNE montage with the desired channels

        Args:
            kept_ch_names (_type_, optional): Channels to be kept. Must be a subset of ch_names. If None, keep all. Defaults to None.
            kind (str, optional): Montage kind. Defaults to "GSN-HydroCel-129". mne.channels.get_builtin_montages() to see the list of available montages.
            head_size (float, optional): Head size for determining channel positions (head scale factor). Defaults to 0.108.

        Returns:
            mne.channels.DigMontage: MNE montage
        """
        original_montage = mne.channels.make_standard_montage(kind, head_size=head_size)  # type: ignore
        original_names_upper = [name.upper() for name in original_montage.ch_names]
        kept_ch_names_upper = (
            [name.upper() for name in kept_ch_names]
            if kept_ch_names is not None
            else None
        )
        new_montage = original_montage.copy()

        ind = [
            i
            for (i, channel) in enumerate(original_names_upper)
            if channel
            in (
                kept_ch_names_upper
                if kept_ch_names is not None
                else original_names_upper
            )
        ]
        # Keep only the desired channels
        new_montage.ch_names = [original_montage.ch_names[x] for x in ind]
        kept_channel_info = [original_montage.dig[3:][x] for x in ind]
        # Keep the first three rows as they are the fiducial points information
        new_montage.dig = new_montage.dig[:3] + kept_channel_info  #

        return new_montage

    @classmethod
    def get_montage(
        cls, montage_name="GSN-HydroCel-129-downsample-111", head_size=0.108
    ) -> mne.channels.DigMontage:
        """Get mne montage given montage name and head size. If the montage name is a downsampled montage, the kept channels are specified in the electrode_names dictionary.
        Montage name should be available in downsampled_montages.py

        Args:
            montage_name (str, optional): Montage name, either mne.channels.get_builtin_montages() or MontageMNE.montage_names Defaults to "GSN-HydroCel-129-downsample-111".
            head_size (float, optional): Head size for determining channel positions (head scale factor). Defaults to 0.108.
        Returns:
            mne.channels.DigMontage: MNE montage with desired channels
        """
        assert (
            montage_name in cls.montage_names
        ), f"Montage {montage_name} not found. Available montages: {montage_names}"
        kind = (
            montage_name
            if "downsample" not in montage_name
            else montage_name.split("-downsample")[0]
        )
        kept_ch_names = None if kind == montage_name else electrode_names[montage_name]
        montage = cls._get_standard_montage(
            kept_ch_names=kept_ch_names, kind=kind, head_size=head_size
        )
        montage = mne.channels.montage.transform_to_head(montage)
        return montage

    @classmethod
    # Create MNE info using sfreq, montage/montage_name
    # Montage can be MNE montage or downsampled version defined in montages.py
    def get_info(cls, sfreq=None, montage=None, **kwargs) -> mne.Info:
        """Get MNe infor object to be used for corregistration. sfreq is dummy data and should be adjusted accordingly if this info object was to be used with real data.

        Args:
            sfreq (float, optional): _description_. Defaults to None.
            montage (mne.channels.DigMontage, optional): MNE montage, if None, create one with kwargs. Defaults to None.

        Returns:
            mne.Info: MNE info object
        """
        if montage is None:
            montage = cls.get_montage(**kwargs)

        if sfreq is None:
            sfreq = 1000
        info = mne.create_info(
            montage.ch_names, sfreq, ch_types=["eeg"] * len(montage.ch_names), verbose=0  # type: ignore
        )
        info.set_montage(montage)
        return info
