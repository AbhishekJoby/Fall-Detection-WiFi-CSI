"""Signal transformations used by the preprocessing pipeline."""

import numpy as np


def select_subcarrier_trace(amplitudes: np.ndarray, subcarrier: int) -> np.ndarray:
    """Return one channel's time series, using the first antenna when present."""
    if subcarrier < 0 or subcarrier >= amplitudes.shape[0]:
        raise ValueError(
            f"Subcarrier {subcarrier} is out of range for {amplitudes.shape[0]} channels"
        )
    if amplitudes.ndim == 3:
        return amplitudes[subcarrier, :, 0]
    if amplitudes.ndim == 2:
        return amplitudes[subcarrier]
    raise ValueError(
        f"Expected amplitudes shaped (channels, time[, antennas]), got {amplitudes.shape}"
    )


def amplitude_to_decibels(amplitudes: np.ndarray) -> np.ndarray:
    """Convert linear amplitude to dB relative to unit amplitude."""
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    return 20 * np.log10(np.maximum(amplitudes, 1e-12))