"""Recording discovery and CSI data loading."""

from pathlib import Path

import h5py
import numpy as np


def find_recording(input_dir: Path, recording_path: Path | None = None) -> Path:
    """Resolve an explicit recording path or select the first HDF5 recording."""
    if recording_path is not None:
        source = recording_path.resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Recording does not exist: {source}")
        return source

    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    recordings = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
    )
    if not recordings:
        raise FileNotFoundError(f"No HDF5 recordings found in {input_dir}")
    return recordings[0]


def load_csi_amplitudes(recording_path: Path) -> np.ndarray:
    """Load and validate CSI_amps shaped (channels, time[, antennas])."""
    with h5py.File(recording_path, "r") as recording:
        if "CSI_amps" not in recording:
            raise KeyError(f"{recording_path} does not contain a root-level CSI_amps dataset")
        amplitudes = recording["CSI_amps"][...]

    if amplitudes.ndim not in (2, 3):
        raise ValueError(
            f"Expected CSI_amps shaped (channels, time[, antennas]), got {amplitudes.shape}"
        )
    if np.any(amplitudes < 0):
        raise ValueError("CSI_amps contains negative values; expected amplitudes")
    return amplitudes