"""Run the CSI median-filtering preview pipeline."""

import argparse
from pathlib import Path

from preprocessing import (
    amplitude_to_decibels,
    apply_median_filter,
    find_recording,
    load_csi_amplitudes,
    select_subcarrier_trace,
)
from visualization import plot_filter_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter CSI amplitudes and plot a before/after comparison in dB."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("FallDetection_DATA"),
        help="Dataset directory (default: FallDetection_DATA)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="HDF5 recording to preview (default: first recording under --input-dir)",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        help="Chart path (default: <input-dir>_median_filter_preview.png)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Odd number of time samples in the median window (default: 5)",
    )
    parser.add_argument(
        "--subcarrier",
        type=int,
        default=0,
        help="Zero-based subcarrier to plot (default: 0)",
    )
    args = parser.parse_args()

    if args.window_size < 1 or args.window_size % 2 == 0:
        parser.error("--window-size must be a positive odd integer")
    if args.subcarrier < 0:
        parser.error("--subcarrier must be zero or greater")
    return args


def run_pipeline(args: argparse.Namespace) -> tuple[Path, Path]:
    """Load, filter, convert, and visualize one recording."""
    source = find_recording(args.input_dir, args.file)
    amplitudes = load_csi_amplitudes(source)
    filtered = apply_median_filter(amplitudes, args.window_size)

    before_trace = select_subcarrier_trace(amplitudes, args.subcarrier)
    after_trace = select_subcarrier_trace(filtered, args.subcarrier)
    before_db = amplitude_to_decibels(before_trace)
    after_db = amplitude_to_decibels(after_trace)

    input_dir = args.input_dir.resolve()
    output_image = (
        args.output_image.resolve()
        if args.output_image
        else input_dir.parent / f"{input_dir.name}_median_filter_preview.png"
    )
    plot_filter_comparison(
        before_db,
        after_db,
        output_image,
        source.name,
        args.subcarrier,
        args.window_size,
    )
    return source, output_image


def main() -> None:
    args = parse_args()
    try:
        source, output_image = run_pipeline(args)
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(f"Recording: {source}")
    print(f"Chart: {output_image}")


if __name__ == "__main__":
    main()