#!/usr/bin/env python3
"""Generate a calibrated, noisy synthetic MWI dataset.

Full Phase 1 pipeline:
  1. Load phantom
  2. Generate tumor scan (gprMax FDTD)
  3. Generate reference scan (same phantom, no tumor)
  4. Calibrate (subtract reference)
  5. Add realistic noise
  6. Export to npz

Usage:
    # Single calibrated scan
    python scripts/generate_dataset.py --phantom 071904 --preset umbmid --tumor-mm 12

    # Multiple scans with varying tumors
    python scripts/generate_dataset.py --phantom 071904 --preset umbmid \
        --n-scans 10 --tumor-range 5 25 --seed 0
"""

import argparse
import os
import time
import numpy as np

from wavecare.synth.phantoms import load_uwcem_phantom
from wavecare.synth.scenes import generate_scene, scene_to_metadata
from wavecare.synth.solver import (
    prepare_2d_geometry,
    write_geometry_files,
    generate_gprmax_inputs,
    run_scan,
    collect_results,
)
from wavecare.synth.export import td_to_fd, calibrate
from wavecare.synth.noise import apply_noise_model, random_noise_params
from wavecare.acqui import presets


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _get_preset(name):
    return {
        "umbmid": presets.umbmid_gen2,
        "maria": presets.maria_m5,
        "mammowave": presets.mammowave,
    }[name]()


def run_single_scan(phantom, array_geo, has_tumor, tumor_params,
                    seed, dx, work_dir, min_clearance_m=0.0):
    """Run one FDTD scan (tumor or reference). Returns results dict."""
    rng = np.random.default_rng(seed)
    scene = generate_scene(phantom, has_tumor=has_tumor, rng=rng,
                           tumor_params=tumor_params)

    geo_info = prepare_2d_geometry(scene, dx=dx)
    write_geometry_files(geo_info, work_dir)
    inputs = generate_gprmax_inputs(
        geo_info,
        array_geo,
        work_dir,
        min_clearance_m=min_clearance_m,
    )

    run_scan(inputs)
    results = collect_results(work_dir, inputs,
                              target_freqs_hz=array_geo.freqs_hz)

    return results, scene, geo_info


def generate_one(phantom, array_geo, tumor_mm, seed, dx,
                 output_dir, snr_db=30.0, scan_id=0, phant_id="SYN",
                 min_clearance_m=0.0):
    """Generate one calibrated, noisy scan.

    Returns dict with all data and metadata.
    """
    rng = np.random.default_rng(seed)

    # --- Tumor scan ---
    tumor_work = os.path.join(output_dir, f"work_tumor_{scan_id:04d}")
    os.makedirs(tumor_work, exist_ok=True)

    tumor_params = {"diameter_mm": tumor_mm, "shape": "sphere"}
    print(f"  [tumor] Running {array_geo.n_measurements()} sims...")
    t0 = time.time()
    tumor_results, tumor_scene, geo_info = run_single_scan(
        phantom, array_geo, has_tumor=True,
        tumor_params=tumor_params, seed=seed, dx=dx,
        work_dir=tumor_work, min_clearance_m=min_clearance_m,
    )
    print(f"  [tumor] Done in {(time.time()-t0)/60:.1f} min")

    # --- Reference scan (same phantom, no tumor, same geometry) ---
    ref_work = os.path.join(output_dir, f"work_ref_{scan_id:04d}")
    os.makedirs(ref_work, exist_ok=True)

    # Use a different seed for reference to avoid correlated randomization,
    # but keep the same phantom (no tumor, no dielectric perturbation)
    ref_seed = seed + 1_000_000
    print(f"  [ref]   Running {array_geo.n_measurements()} sims...")
    t0 = time.time()
    ref_results, _, _ = run_single_scan(
        phantom, array_geo, has_tumor=False,
        tumor_params=None, seed=ref_seed, dx=dx,
        work_dir=ref_work, min_clearance_m=min_clearance_m,
    )
    print(f"  [ref]   Done in {(time.time()-t0)/60:.1f} min")

    # --- Calibrate ---
    fd_tumor = tumor_results["fd_data"]
    fd_ref = ref_results["fd_data"]
    fd_cal = calibrate(fd_tumor, fd_ref)
    print(f"  [cal]   Calibrated: max |S_cal| = {np.max(np.abs(fd_cal)):.2e}")

    # --- Add noise ---
    noise_params = random_noise_params(rng=rng)
    fd_noisy = apply_noise_model(fd_cal, rng=rng, **noise_params)
    print(f"  [noise] SNR={noise_params['snr_db']:.1f} dB, "
          f"phase={noise_params['phase_std_deg']:.1f} deg, "
          f"amp={noise_params['amp_std_db']:.2f} dB")

    # --- Metadata ---
    metadata = scene_to_metadata(tumor_scene, scan_id=scan_id,
                                 phant_id=phant_id)
    metadata["seed"] = seed
    metadata["dx"] = dx
    metadata["ant_radius_cm"] = array_geo.radius_m * 100.0
    metadata["min_clearance_mm"] = min_clearance_m * 1000.0
    metadata["preset"] = "custom"
    metadata["noise_snr_db"] = noise_params["snr_db"]
    metadata["noise_phase_deg"] = noise_params["phase_std_deg"]
    metadata["noise_amp_db"] = noise_params["amp_std_db"]
    metadata["slice_idx"] = geo_info["slice_idx"]

    return {
        "td_tumor": tumor_results["td_data"],
        "td_ref": ref_results["td_data"],
        "fd_tumor": fd_tumor,
        "fd_ref": fd_ref,
        "fd_cal": fd_cal,
        "fd_noisy": fd_noisy,
        "freqs_hz": tumor_results["freqs_hz"],
        "dt": tumor_results["dt"],
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate calibrated, noisy synthetic MWI dataset")
    parser.add_argument("--phantom", required=True,
                        help="UWCEM phantom ID (e.g., 071904)")
    parser.add_argument("--preset", default="umbmid",
                        choices=["umbmid", "maria", "mammowave"])
    parser.add_argument("--n-scans", type=int, default=1,
                        help="Number of scans to generate")
    parser.add_argument("--tumor-mm", type=float, default=12.0,
                        help="Fixed tumor diameter (used if --n-scans 1)")
    parser.add_argument("--tumor-range", type=float, nargs=2, default=None,
                        metavar=("MIN", "MAX"),
                        help="Tumor diameter range in mm (for multi-scan)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dx", type=float, default=0.001)
    parser.add_argument("--radius-cm", type=float, default=None,
                        help="Override antenna ring radius (cm)")
    parser.add_argument("--min-clearance-mm", type=float, default=0.0,
                        help="Required antenna-to-tissue clearance (mm)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(_PROJECT_ROOT, "data", "uwcem")
    output_dir = args.output_dir or os.path.join(
        _PROJECT_ROOT, "output", f"dataset_{args.phantom}_{args.preset}")
    os.makedirs(output_dir, exist_ok=True)

    array_geo = _get_preset(args.preset)
    if args.radius_cm is not None:
        array_geo.radius_m = args.radius_cm / 100.0

    print("=" * 60)
    print("WaveCare Dataset Generation (Phase 1)")
    print(f"  Phantom: {args.phantom}")
    print(f"  Preset:  {args.preset} ({array_geo.n_antennas} ant, "
          f"{array_geo.mode})")
    print(f"  Scans:   {args.n_scans}")
    print(f"  Cell:    {args.dx*1000:.1f} mm")
    print(f"  Radius:  {array_geo.radius_m*100:.2f} cm")
    print(f"  Output:  {output_dir}")
    print("=" * 60)

    # Load phantom
    print("\nLoading phantom...")
    phantom = load_uwcem_phantom(os.path.join(data_dir, args.phantom))
    print(f"  Shape: {phantom['shape']}, voxel: {phantom['voxel_mm']} mm")

    # Determine tumor sizes
    rng = np.random.default_rng(args.seed)
    if args.n_scans == 1:
        tumor_sizes = [args.tumor_mm]
    elif args.tumor_range:
        tumor_sizes = rng.uniform(
            args.tumor_range[0], args.tumor_range[1], args.n_scans)
    else:
        tumor_sizes = rng.uniform(5.0, 25.0, args.n_scans)

    # Per-scan seeds
    scan_seeds = rng.integers(0, 2**31, size=args.n_scans)

    # Generate all scans
    all_fd_cal = []
    all_fd_noisy = []
    all_metadata = []

    t_total = time.time()
    for i in range(args.n_scans):
        print(f"\n--- Scan {i+1}/{args.n_scans}: "
              f"tumor={tumor_sizes[i]:.1f}mm, seed={scan_seeds[i]} ---")

        result = generate_one(
            phantom, array_geo,
            tumor_mm=float(tumor_sizes[i]),
            seed=int(scan_seeds[i]),
            dx=args.dx,
            output_dir=output_dir,
            scan_id=i,
            phant_id=args.phantom,
            min_clearance_m=args.min_clearance_mm / 1000.0,
        )

        all_fd_cal.append(result["fd_cal"])
        all_fd_noisy.append(result["fd_noisy"])
        all_metadata.append(result["metadata"])

        # Save individual scan
        scan_path = os.path.join(output_dir, f"scan_{i:04d}.npz")
        np.savez_compressed(
            scan_path,
            fd_cal=result["fd_cal"],
            fd_noisy=result["fd_noisy"],
            td_tumor=result["td_tumor"],
            td_ref=result["td_ref"],
            freqs_hz=result["freqs_hz"],
            dt=result["dt"],
            **{f"meta_{k}": v for k, v in result["metadata"].items()},
        )
        print(f"  Saved: {scan_path}")

    # Save combined dataset
    n_meas, n_freqs = all_fd_noisy[0].shape
    fd_dataset = np.stack(all_fd_noisy)  # (n_scans, n_meas, n_freqs)
    fd_cal_dataset = np.stack(all_fd_cal)
    labels = np.array([1] * args.n_scans)  # all have tumors

    dataset_path = os.path.join(output_dir, "dataset.npz")
    np.savez_compressed(
        dataset_path,
        fd_data=fd_dataset,
        fd_cal=fd_cal_dataset,
        labels=labels,
        freqs_hz=result["freqs_hz"],
        n_scans=args.n_scans,
        n_measurements=n_meas,
        n_freqs=n_freqs,
        phantom_id=args.phantom,
        preset=args.preset,
    )

    elapsed = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"Dataset complete: {args.n_scans} scans in {elapsed:.1f} min")
    print(f"  Shape: {fd_dataset.shape}")
    print(f"  Saved: {dataset_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
