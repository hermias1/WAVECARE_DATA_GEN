#!/usr/bin/env python3
"""Numerical sensitivity report for spatial resolution and bandwidth.

Computes wavelength and cells-per-wavelength across tissues and frequencies.
Useful to decide dx/fmax tradeoffs before large runs.
"""

import argparse
import math
import os
import sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wavecare.synth.dielectrics import get_tissue_permittivity


_C0 = 299_792_458.0

_DEFAULT_TISSUES = [
    "skin_dry",
    "adipose_low",
    "adipose_med",
    "adipose_high",
    "transitional_med",
    "fibroglandular_low",
    "fibroglandular_med",
    "fibroglandular_high",
    "tumor_malignant",
]


def _cells_per_wavelength(freq_hz, eps_r, dx_m):
    lam = _C0 / (freq_hz * math.sqrt(max(eps_r, 1e-12)))
    return lam / dx_m, lam


def run_report(
    dx_mm=1.0,
    fmin_ghz=1.0,
    fmax_ghz=8.0,
    n_freqs=8,
    model="cole_cole",
    min_cells=10.0,
    tissues=None,
):
    tissues = tissues or list(_DEFAULT_TISSUES)
    dx_m = dx_mm / 1000.0
    freqs_hz = np.linspace(fmin_ghz * 1e9, fmax_ghz * 1e9, n_freqs)

    print("=" * 72)
    print("WaveCare Numerical Sensitivity Report")
    print(f"  dx = {dx_mm:.3f} mm")
    print(f"  band = {fmin_ghz:.2f} - {fmax_ghz:.2f} GHz ({n_freqs} samples)")
    print(f"  dielectric model = {model}")
    print(f"  target minimum cells/lambda = {min_cells:.1f}")
    print("=" * 72)

    per_tissue = {}
    for tissue in tissues:
        eps = get_tissue_permittivity(tissue, freqs_hz, model=model)
        eps_r = np.real(eps)
        cells = []
        lambdas_mm = []
        for f, er in zip(freqs_hz, eps_r):
            cpl, lam = _cells_per_wavelength(float(f), float(er), dx_m)
            cells.append(cpl)
            lambdas_mm.append(lam * 1000.0)
        per_tissue[tissue] = {
            "eps_r": eps_r,
            "cells": np.array(cells),
            "lambda_mm": np.array(lambdas_mm),
        }

    print("\nWorst-case at top frequency:")
    print("  Tissue                 eps_r(fmax)   lambda(mm)   cells/lambda")
    print("  --------------------   ----------    ----------   ------------")
    worst_cpl = float("inf")
    worst_tissue = None
    for tissue in tissues:
        er = float(per_tissue[tissue]["eps_r"][-1])
        lam_mm = float(per_tissue[tissue]["lambda_mm"][-1])
        cpl = float(per_tissue[tissue]["cells"][-1])
        if cpl < worst_cpl:
            worst_cpl = cpl
            worst_tissue = tissue
        print(f"  {tissue:<20} {er:>10.2f} {lam_mm:>12.2f} {cpl:>14.2f}")

    print("\nBand-min cells/lambda (across all sampled freqs):")
    for tissue in tissues:
        min_cpl = float(np.min(per_tissue[tissue]["cells"]))
        print(f"  {tissue:<20} {min_cpl:>6.2f}")

    # Conservative fmax estimate from worst eps_r at top frequency.
    worst_eps = float(np.max([per_tissue[t]["eps_r"][-1] for t in tissues]))
    rec_fmax_hz = _C0 / (dx_m * min_cells * math.sqrt(max(worst_eps, 1e-12)))
    rec_fmax_ghz = rec_fmax_hz / 1e9
    print("\nRecommendation:")
    print(f"  Worst tissue at fmax: {worst_tissue} ({worst_cpl:.2f} cells/lambda)")
    print(f"  To keep >= {min_cells:.1f} cells/lambda (conservative): "
          f"fmax <= {rec_fmax_ghz:.2f} GHz")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Resolution vs frequency sensitivity")
    parser.add_argument("--dx-mm", type=float, default=1.0)
    parser.add_argument("--fmin-ghz", type=float, default=1.0)
    parser.add_argument("--fmax-ghz", type=float, default=8.0)
    parser.add_argument("--n-freqs", type=int, default=8)
    parser.add_argument("--model", choices=["cole_cole", "debye"], default="cole_cole")
    parser.add_argument("--min-cells", type=float, default=10.0)
    parser.add_argument("--tissues", nargs="*", default=None,
                        help="Optional tissue names from wavecare.synth.dielectrics")
    args = parser.parse_args()

    run_report(
        dx_mm=args.dx_mm,
        fmin_ghz=args.fmin_ghz,
        fmax_ghz=args.fmax_ghz,
        n_freqs=args.n_freqs,
        model=args.model,
        min_cells=args.min_cells,
        tissues=args.tissues,
    )


if __name__ == "__main__":
    main()
