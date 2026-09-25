"""Temporal median filtering for CSI amplitude arrays."""

import numpy as np
from scipy.ndimage import median_filter


def apply_median_filter(amplitudes: np.ndarray, window_size: int) -> np.ndarray:
    """Apply a centered median filter over time independently per channel."""
    if amplitudes.ndim not in (2, 3):
        raise ValueError(
            f"Expected amplitudes shaped (channels, time[, antennas]), got {amplitudes.shape}"
        )
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    filter_size = [1] * amplitudes.ndim
    filter_size[-2] = window_size
    return median_filter(amplitudes, size=filter_size, mode="nearest")