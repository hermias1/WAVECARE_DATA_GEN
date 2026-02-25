#!/usr/bin/env python3
"""Compare one synthetic scan with one real UM-BMID scan.

This script computes lightweight similarity metrics for quick sanity checks.
"""

import argparse
import json
import os
import pickle
import sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wavecare.eval.compare import (
    signal_statistics,
    spectral_similarity,
    angular_variation_similarity,
)


def _load_complex_array(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        d = np.load(path)  # nosec - local trusted files
        for key in ["fd_data", "fd_noisy", "fd_cal", "fd_s11"]:
            if key in d:
                return d[key], d
        raise ValueError(f"No supported FD key found in {path}")
    if ext in [".pickle", ".pkl"]:
        with open(path, "rb") as f:
            arr = pickle.load(f)  # nosec B301 - review script for local files
        return np.asarray(arr), None
    raise ValueError(f"Unsupported file extension: {ext}")


def _select_scan(arr, scan_idx):
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if scan_idx < 0 or scan_idx >= arr.shape[0]:
            raise IndexError(f"scan_idx={scan_idx} out of range [0, {arr.shape[0]-1}]")
        return arr[scan_idx]
    raise ValueError(f"Expected 2D or 3D complex data, got shape {arr.shape}")


def compare(synth_path, real_path, synth_idx=0, real_idx=0):
    synth_arr, synth_meta = _load_complex_array(synth_path)
    real_arr, real_meta = _load_complex_array(real_path)

    synth_fd = _select_scan(synth_arr, synth_idx)
    real_fd = _select_scan(real_arr, real_idx)

    n_meas = min(synth_fd.shape[0], real_fd.shape[0])
    n_freqs = min(synth_fd.shape[1], real_fd.shape[1])
    synth_fd = synth_fd[:n_meas, :n_freqs]
    real_fd = real_fd[:n_meas, :n_freqs]

    if synth_meta is not None and "freqs_hz" in synth_meta:
        freqs = np.asarray(synth_meta["freqs_hz"])[:n_freqs]
    elif real_meta is not None and "freqs_hz" in real_meta:
        freqs = np.asarray(real_meta["freqs_hz"])[:n_freqs]
    else:
        freqs = np.linspace(1e9, 8e9, n_freqs)

    report = {
        "inputs": {
            "synthetic_path": synth_path,
            "real_path": real_path,
            "synthetic_index": synth_idx,
            "real_index": real_idx,
            "aligned_shape": [int(n_meas), int(n_freqs)],
        },
        "stats": {
            "synthetic": signal_statistics(synth_fd),
            "real": signal_statistics(real_fd),
        },
        "spectral_similarity": spectral_similarity(synth_fd, real_fd, freqs),
        "angular_similarity": angular_variation_similarity(synth_fd, real_fd),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare synthetic vs UM-BMID scan")
    parser.add_argument("--synthetic", required=True,
                        help="Path to synthetic .npz/.pickle")
    parser.add_argument("--real", required=True,
                        help="Path to UM-BMID .pickle or .npz")
    parser.add_argument("--synthetic-index", type=int, default=0)
    parser.add_argument("--real-index", type=int, default=0)
    parser.add_argument("--output", default=None,
                        help="Optional path to save report JSON")
    args = parser.parse_args()

    rep = compare(
        synth_path=args.synthetic,
        real_path=args.real,
        synth_idx=args.synthetic_index,
        real_idx=args.real_index,
    )

    print(json.dumps(rep, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
