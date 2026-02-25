#!/usr/bin/env python3
"""Generate a synthetic MWI scan from a UWCEM phantom.

Usage:
    python scripts/generate_scan.py --phantom 071904 --preset umbmid
    python scripts/generate_scan.py --phantom 062204 --preset maria --no-tumor
    python scripts/generate_scan.py --phantom 071904 --tumor-mm 8 --seed 123
"""

import argparse
import os
import time
import numpy as np
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wavecare.synth.phantoms import load_uwcem_phantom
from wavecare.synth.scenes import generate_scene
from wavecare.synth.solver import (
    prepare_2d_geometry,
    write_geometry_files,
    generate_gprmax_inputs,
    run_scan,
    collect_results,
)
from wavecare.acqui import presets


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic MWI scan")
    parser.add_argument("--phantom", required=True,
                        help="UWCEM phantom ID (e.g., 071904)")
    parser.add_argument("--preset", default="umbmid",
                        choices=["umbmid", "maria", "mammowave"],
                        help="Acquisition geometry preset")
    parser.add_argument("--tumor-mm", type=float, default=12.0,
                        help="Tumor diameter in mm (0 = no tumor)")
    parser.add_argument("--no-tumor", action="store_true",
                        help="Generate scan without tumor")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--dx", type=float, default=0.001,
                        help="Cell size in meters (default: 0.001 = 1mm)")
    parser.add_argument("--radius-cm", type=float, default=None,
                        help="Override antenna ring radius (cm)")
    parser.add_argument("--min-clearance-mm", type=float, default=0.0,
                        help="Required antenna-to-tissue clearance (mm)")
    parser.add_argument("--center-mode", default="auto",
                        choices=["auto", "volume", "tissue_centroid",
                                 "skin_centroid", "ring_fit"],
                        help="How to place array center relative to phantom")
    parser.add_argument("--center-search-mm", type=float, default=20.0,
                        help="Search half-width for ring_fit (mm)")
    parser.add_argument("--data-dir", default=None,
                        help="Path to UWCEM phantom data")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory")
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(_PROJECT_ROOT, "data", "uwcem")
    output_dir = args.output_dir or os.path.join(
        _PROJECT_ROOT, "output", f"scan_{args.phantom}_{args.preset}")

    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, "gprmax_work")
    os.makedirs(work_dir, exist_ok=True)

    # Acquisition geometry
    if args.preset == "umbmid":
        array_geo = presets.umbmid_gen2()
    elif args.preset == "maria":
        array_geo = presets.maria_m5()
    elif args.preset == "mammowave":
        array_geo = presets.mammowave()
    if args.radius_cm is not None:
        array_geo.radius_m = args.radius_cm / 100.0

    has_tumor = not args.no_tumor and args.tumor_mm > 0

    print("=" * 60)
    print(f"WaveCare Synthetic Scan Generation")
    print(f"  Phantom: {args.phantom}")
    print(f"  Preset:  {args.preset} ({array_geo.n_antennas} antennas, "
          f"{array_geo.mode})")
    print(f"  Tumor:   {'%.1f mm' % args.tumor_mm if has_tumor else 'none'}")
    print(f"  Cell:    {args.dx*1000:.1f} mm")
    print(f"  Radius:  {array_geo.radius_m*100:.2f} cm")
    print(f"  Center:  {args.center_mode}")
    print(f"  Output:  {output_dir}")
    print("=" * 60)

    # Load phantom
    print("\n[1/5] Loading phantom...")
    phantom_dir = os.path.join(data_dir, args.phantom)
    phantom = load_uwcem_phantom(phantom_dir)
    print(f"  Shape: {phantom['shape']}, voxel: {phantom['voxel_mm']} mm")

    # Generate scene
    print("\n[2/5] Generating scene...")
    rng = np.random.default_rng(args.seed)
    tumor_params = {"diameter_mm": args.tumor_mm, "shape": "sphere"} if has_tumor else None
    scene = generate_scene(phantom, has_tumor=has_tumor, rng=rng,
                           tumor_params=tumor_params)
    if scene.get("tumor_info"):
        ti = scene["tumor_info"]
        print(f"  Tumor: {ti['diameter_mm']:.1f} mm at "
              f"({ti['tum_x']:.1f}, {ti['tum_y']:.1f}, {ti['tum_z']:.1f}) cm")

    # Prepare geometry
    print("\n[3/5] Preparing 2D geometry...")
    geo_info = prepare_2d_geometry(scene, dx=args.dx)
    write_geometry_files(geo_info, work_dir)
    print(f"  Domain: {geo_info['domain_m'][0]*100:.1f} x "
          f"{geo_info['domain_m'][1]*100:.1f} cm")
    print(f"  Slice: z={geo_info['slice_idx']}")

    # Generate input files
    print(f"\n[4/5] Generating {array_geo.n_measurements()} input files...")
    inputs = generate_gprmax_inputs(
        geo_info,
        array_geo,
        work_dir,
        min_clearance_m=args.min_clearance_mm / 1000.0,
        center_mode=args.center_mode,
        center_search_m=args.center_search_mm / 1000.0,
    )
    print(f"  Generated {len(inputs)} simulation configs")
    if "array_center_shift_mm" in geo_info:
        sx, sy = geo_info["array_center_shift_mm"]
        print(f"  Center shift: ({sx:+.1f}, {sy:+.1f}) mm")

    # Run
    print(f"\n[5/5] Running simulations...")
    total_time = run_scan(inputs)
    print(f"  Completed in {total_time/60:.1f} minutes")

    # Collect results
    print("\nCollecting results...")
    results = collect_results(work_dir, inputs,
                              target_freqs_hz=array_geo.freqs_hz)
    print(f"  TD data: {results['td_data'].shape}")
    print(f"  FD data: {results['fd_data'].shape}")

    # Save
    metadata = {
        "phantom_id": args.phantom,
        "preset": args.preset,
        "has_tumor": has_tumor,
        "tumor_mm": args.tumor_mm if has_tumor else 0,
        "seed": args.seed,
        "dx": args.dx,
        "center_mode": args.center_mode,
        "center_search_mm": args.center_search_mm,
        "n_measurements": len(inputs),
        "slice_idx": geo_info["slice_idx"],
    }
    if "array_center_m" in geo_info:
        metadata["array_center_m"] = geo_info["array_center_m"]
    if "array_center_shift_mm" in geo_info:
        metadata["array_center_shift_mm"] = geo_info["array_center_shift_mm"]
    if scene.get("tumor_info"):
        for k, v in scene["tumor_info"].items():
            metadata[f"tumor_{k}"] = v

    np.savez_compressed(
        os.path.join(output_dir, "scan_data.npz"),
        td_data=results["td_data"],
        fd_data=results["fd_data"],
        dt=results["dt"],
        freqs_hz=results["freqs_hz"],
        **{f"meta_{k}": v for k, v in metadata.items()},
    )
    print(f"\nSaved: {output_dir}/scan_data.npz")
    print("Done!")


if __name__ == "__main__":
    main()
