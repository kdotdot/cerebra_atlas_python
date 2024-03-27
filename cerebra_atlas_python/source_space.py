import logging
import mne
import os.path as op
import numpy as np


logger = logging.getLogger(__name__)

def get_source_space_mask(cerebra, source_space_grid_size, include_non_cortical, include_wm):
    # Create grid (downsample available volume)
    size = source_space_grid_size
    grid = np.zeros((256, 256, 256))
    a = np.arange(size - 1, 256, size)
    for x in a:
        for y in a:
            grid[x, y, a] = 1
    grid_mask = grid.astype(bool)

    # If include non-cortical keep all
    if include_non_cortical:
        not_zero_mask = cerebra.cerebra_volume != cerebra.get_region_id_from_region_name(
            "Empty"
        )
        not_wm_mask = cerebra.cerebra_volume != cerebra.get_region_id_from_region_name(
            "White matter"
        )
        downsampled_not_zero_mask = np.logical_and(grid_mask, not_zero_mask)
        downsampled_not_wm_mask = np.logical_and(grid_mask, not_wm_mask)
        # Combined mask is not zero and not wm
        combined_mask = np.logical_and(
            downsampled_not_zero_mask, downsampled_not_wm_mask
        )

    else:  # Keep only cortical:
        # Get cortical region ids
        cortical_ids = cerebra.label_details[cerebra.label_details["cortical"] == True][
            "CerebrA ID"
        ].values
        combined_mask = np.zeros((256, 256, 256)).astype(bool)
        for c_id in cortical_ids:
            # Handle each region individually
            region_mask = cerebra.cerebra_volume == c_id
            # Downsample based on grid
            downsampled_region_mask = np.logical_and(grid_mask, region_mask)
            # Add region to mask
            combined_mask = np.logical_or(combined_mask, downsampled_region_mask)

    # Add whitematter if needed
    if include_wm:
        whitematter_mask = (
            cerebra.cerebra_volume
            == cerebra.get_region_id_from_region_name("White matter")
        )
        # Downsample
        downsampled_whitematter_mask = np.logical_and(grid_mask, whitematter_mask)
        combined_mask = np.logical_or(combined_mask, downsampled_whitematter_mask)

    return combined_mask

