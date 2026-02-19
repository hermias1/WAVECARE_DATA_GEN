#!/usr/bin/env python3
"""Verify a generated 3D MWI dataset.

Checks:
  - Shape and dtype
  - No NaN/Inf
  - Label distribution
  - Signal energy: positive > negative (on average)
  - Frequency range

Usage:
    python verify_dataset.py /path/to/dataset_3d.npz
"""

import argparse
import sys
import numpy as np


def verify(path):
    print(f"Loading {path}...")
    d = np.load(path)  # nosec — our own generated .npz files

    keys = list(d.keys())
    print(f"  Keys: {keys}")

    errors = []

    # --- fd_data ---
    if "fd_data" not in d:
        errors.append("Missing key: fd_data")
        print(f"FAIL: {errors[-1]}")
        return errors

    fd = d["fd_data"]
    print(f"  fd_data:  shape={fd.shape}, dtype={fd.dtype}")

    if fd.ndim != 3:
        errors.append(f"fd_data should be 3D, got {fd.ndim}D")

    if not np.iscomplexobj(fd):
        errors.append(f"fd_data should be complex, got {fd.dtype}")

    if not np.all(np.isfinite(fd)):
        n_bad = np.sum(~np.isfinite(fd))
        errors.append(f"fd_data has {n_bad} NaN/Inf values")

    if np.max(np.abs(fd)) == 0:
        errors.append("fd_data is all zeros")

    # --- labels ---
    if "labels" in d:
        labels = d["labels"]
        print(f"  labels:   shape={labels.shape}, values={np.unique(labels)}")
        n_pos = int(np.sum(labels == 1))
        n_neg = int(np.sum(labels == 0))
        print(f"  Distribution: {n_pos} positive, {n_neg} negative")

        if n_pos == 0:
            errors.append("No positive samples (label=1)")
        if n_neg == 0:
            errors.append("No negative samples (label=0)")

        # Energy check
        if n_pos > 0 and n_neg > 0:
            pos_energy = np.mean(np.abs(fd[labels == 1]) ** 2)
            neg_energy = np.mean(np.abs(fd[labels == 0]) ** 2)
            ratio = pos_energy / neg_energy if neg_energy > 0 else float('inf')
            print(f"  Energy:   positive={pos_energy:.2e}, "
                  f"negative={neg_energy:.2e}, ratio={ratio:.2f}")
            if pos_energy <= neg_energy:
                errors.append(
                    "Negative samples have >= energy than positive — suspicious")
    else:
        print("  labels:   NOT PRESENT")

    # --- freqs ---
    if "freqs_hz" in d:
        freqs = d["freqs_hz"]
        print(f"  freqs_hz: {freqs[0]/1e9:.2f} - {freqs[-1]/1e9:.2f} GHz "
              f"({len(freqs)} points)")
    else:
        print("  freqs_hz: NOT PRESENT")

    # --- fd_cal ---
    if "fd_cal" in d:
        fd_cal = d["fd_cal"]
        print(f"  fd_cal:   shape={fd_cal.shape}, dtype={fd_cal.dtype}")

    # --- metadata ---
    for key in ["n_scans", "phantom_id", "preset", "mode"]:
        if key in d:
            print(f"  {key}: {d[key]}")

    # --- Dynamic range ---
    if np.max(np.abs(fd)) > 0:
        dr = 20 * np.log10(np.max(np.abs(fd)) / np.mean(np.abs(fd)))
        print(f"  Dynamic range: {dr:.1f} dB")

    # --- Summary ---
    print()
    if errors:
        print(f"FAILED: {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return errors
    else:
        print("ALL CHECKS PASSED")
        return []


def main():
    parser = argparse.ArgumentParser(description="Verify MWI dataset")
    parser.add_argument("dataset", help="Path to .npz dataset file")
    args = parser.parse_args()

    errors = verify(args.dataset)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
