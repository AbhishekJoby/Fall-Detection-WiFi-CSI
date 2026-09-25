"""Visualization helpers for CSI preprocessing results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_filter_comparison(
    before_db: np.ndarray,
    after_db: np.ndarray,
    output_image: Path,
    recording_name: str,
    subcarrier: int,
    window_size: int,
) -> None:
    """Save a line chart comparing one CSI trace before and after filtering."""
    output_image.parent.mkdir(parents=True, exist_ok=True)
    plot_length = min(200, before_db.size, after_db.size)
    time_samples = np.arange(plot_length)
    before_plot = before_db[:plot_length]
    after_plot = after_db[:plot_length]

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.plot(time_samples, before_plot, label="Before median filter", alpha=0.7)
    axis.plot(time_samples, after_plot, label="After median filter", linewidth=1.8)

    combined_db = np.concatenate((before_plot, after_plot))
    visible_db = combined_db[np.isfinite(combined_db) & (combined_db > -239.999)]
    if visible_db.size == 0:
        visible_db = combined_db[np.isfinite(combined_db)]
    if visible_db.size:
        lower = float(visible_db.min())
        upper = float(visible_db.max())
        padding = max((upper - lower) * 0.08, 0.5)
        axis.set_ylim(lower - padding, upper + padding)

    axis.set_xlabel("Time sample")
    axis.set_ylabel("CSI amplitude (dB)")
    axis.set_title(
        f"Subcarrier {subcarrier} | window size {window_size} | {recording_name}"
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_image, dpi=160)
    plt.close(figure)