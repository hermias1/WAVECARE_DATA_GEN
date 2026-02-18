#!/usr/bin/env python3
"""SLURM-based 3D scan generation for DGX cluster.

Three-phase workflow:
  1. prepare  — load phantom, generate .in files (CPU, fast)
  2. submit   — sbatch array job, 1 GPU per simulation
  3. collect  — gather results, calibrate, add noise, export

Usage:
    # Phase 1: prepare geometry + input files
    python slurm_scan.py prepare \
        --phantom 071904 --preset umbmid --tumor-mm 12 \
        --work-dir /scratch/$USER/wavecare/scan_001

    # Phase 2: submit to SLURM
    python slurm_scan.py submit \
        --work-dir /scratch/$USER/wavecare/scan_001 \
        --partition gpu --gres gpu:1

    # Phase 3: collect + calibrate + export
    python slurm_scan.py collect \
        --work-dir /scratch/$USER/wavecare/scan_001 \
        --output-dir /results/scan_001.npz

    # Or run all phases (useful for testing on a single node):
    python slurm_scan.py run-local \
        --phantom 071904 --preset umbmid --tumor-mm 12 \
        --work-dir /tmp/wavecare_test --gpu 0
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np

# Add project root to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wavecare.synth.phantoms import load_uwcem_phantom
from wavecare.synth.scenes import generate_scene, scene_to_metadata
from wavecare.synth.solver import (
    prepare_3d_geometry,
    write_geometry_files,
    generate_gprmax_inputs_3d,
    collect_results,
)
from wavecare.synth.export import calibrate
from wavecare.synth.noise import apply_noise_model, random_noise_params
from wavecare.acqui import presets


def _get_preset(name):
    return {
        "umbmid": presets.umbmid_gen2,
        "maria": presets.maria_m5,
        "mammowave": presets.mammowave,
    }[name]()


# --------------------------------------------------------------------------- #
# Phase 1: prepare
# --------------------------------------------------------------------------- #

def cmd_prepare(args):
    """Load phantom, generate scenes, write geometry + .in files."""
    data_dir = args.data_dir or os.path.join(_PROJECT_ROOT, "data", "uwcem")
    array_geo = _get_preset(args.preset)

    print("Loading phantom...")
    phantom = load_uwcem_phantom(os.path.join(data_dir, args.phantom))
    print(f"  Shape: {phantom['shape']}, voxel: {phantom['voxel_mm']} mm")

    has_tumor = not args.no_tumor and args.tumor_mm > 0
    rng = np.random.default_rng(args.seed)

    # --- Tumor scene ---
    tumor_work = os.path.join(args.work_dir, "tumor")
    os.makedirs(tumor_work, exist_ok=True)

    tumor_params = {"diameter_mm": args.tumor_mm, "shape": "sphere"} if has_tumor else None
    tumor_scene = generate_scene(
        phantom, has_tumor=has_tumor, rng=rng, tumor_params=tumor_params)

    print("Preparing 3D geometry (tumor)...")
    tumor_geo = prepare_3d_geometry(tumor_scene, dx=args.dx, pad_cells=args.pad)
    write_geometry_files(tumor_geo, tumor_work)
    tumor_inputs = generate_gprmax_inputs_3d(tumor_geo, array_geo, tumor_work)
    print(f"  Domain: {tumor_geo['geo_shape']} = "
          f"{np.prod(tumor_geo['geo_shape'])/1e6:.1f}M cells")
    print(f"  {len(tumor_inputs)} .in files written")

    # --- Reference scene (no tumor) ---
    ref_work = os.path.join(args.work_dir, "ref")
    os.makedirs(ref_work, exist_ok=True)

    ref_rng = np.random.default_rng(args.seed + 1_000_000)
    ref_scene = generate_scene(phantom, has_tumor=False, rng=ref_rng)

    print("Preparing 3D geometry (reference)...")
    ref_geo = prepare_3d_geometry(ref_scene, dx=args.dx, pad_cells=args.pad)
    write_geometry_files(ref_geo, ref_work)
    ref_inputs = generate_gprmax_inputs_3d(ref_geo, array_geo, ref_work)
    print(f"  {len(ref_inputs)} .in files written")

    # Save scan metadata
    meta = {
        "phantom_id": args.phantom,
        "preset": args.preset,
        "has_tumor": has_tumor,
        "tumor_mm": args.tumor_mm if has_tumor else 0,
        "seed": args.seed,
        "dx": args.dx,
        "pad_cells": args.pad,
        "n_pairs_tumor": len(tumor_inputs),
        "n_pairs_ref": len(ref_inputs),
        "geo_shape": list(tumor_geo["geo_shape"]),
        "domain_m": list(tumor_geo["domain_m"]),
        "n_cells": int(np.prod(tumor_geo["geo_shape"])),
    }
    if tumor_scene.get("tumor_info"):
        ti = tumor_scene["tumor_info"]
        for k, v in ti.items():
            meta[f"tumor_{k}"] = v if not isinstance(v, np.ndarray) else v.tolist()

    # Scene metadata for export
    meta["scene_metadata"] = scene_to_metadata(
        tumor_scene, scan_id=0, phant_id=args.phantom)

    meta_path = os.path.join(args.work_dir, "scan_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\nPrepare complete. Metadata: {meta_path}")
    print(f"  Total sims: {len(tumor_inputs) + len(ref_inputs)}")
    print(f"  Estimated GPU time: ~{(len(tumor_inputs)+len(ref_inputs))*2/60:.0f} min "
          f"(1 GPU) or ~{(len(tumor_inputs)+len(ref_inputs))*2/60/8:.0f} min (8 GPUs)")


# --------------------------------------------------------------------------- #
# Phase 2: submit SLURM
# --------------------------------------------------------------------------- #

def cmd_submit(args):
    """Submit SLURM array jobs for tumor and reference scans."""
    meta_path = os.path.join(args.work_dir, "scan_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    n_tumor = meta["n_pairs_tumor"]
    n_ref = meta["n_pairs_ref"]

    slurm_script = os.path.join(_SCRIPT_DIR, "slurm_sim.sh")

    # Submit tumor array
    tumor_cmd = [
        "sbatch",
        f"--array=0-{n_tumor-1}",
        f"--partition={args.partition}",
        f"--gres={args.gres}",
        f"--job-name=wc_tumor_{meta['phantom_id']}",
        f"--output={args.work_dir}/tumor/slurm_%A_%a.log",
        f"--time={args.time_limit}",
        slurm_script,
        os.path.join(args.work_dir, "tumor"),
    ]
    if args.account:
        tumor_cmd.insert(1, f"--account={args.account}")

    print(f"Submitting tumor array ({n_tumor} tasks)...")
    print(f"  {' '.join(tumor_cmd)}")
    if not args.dry_run:
        result = subprocess.run(tumor_cmd, capture_output=True, text=True)
        print(f"  {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)
            return

    # Submit reference array
    ref_cmd = [
        "sbatch",
        f"--array=0-{n_ref-1}",
        f"--partition={args.partition}",
        f"--gres={args.gres}",
        f"--job-name=wc_ref_{meta['phantom_id']}",
        f"--output={args.work_dir}/ref/slurm_%A_%a.log",
        f"--time={args.time_limit}",
        slurm_script,
        os.path.join(args.work_dir, "ref"),
    ]
    if args.account:
        ref_cmd.insert(1, f"--account={args.account}")

    print(f"Submitting reference array ({n_ref} tasks)...")
    print(f"  {' '.join(ref_cmd)}")
    if not args.dry_run:
        result = subprocess.run(ref_cmd, capture_output=True, text=True)
        print(f"  {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.strip()}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Phase 3: collect
# --------------------------------------------------------------------------- #

def cmd_collect(args):
    """Gather simulation outputs, calibrate, add noise, export."""
    meta_path = os.path.join(args.work_dir, "scan_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    array_geo = _get_preset(meta["preset"])

    # Reconstruct input lists (same order as prepare)
    tumor_work = os.path.join(args.work_dir, "tumor")
    ref_work = os.path.join(args.work_dir, "ref")

    tumor_inputs = [
        (os.path.join(tumor_work, f"pair_{i:04d}.in"), -1, -1)
        for i in range(meta["n_pairs_tumor"])
    ]
    ref_inputs = [
        (os.path.join(ref_work, f"pair_{i:04d}.in"), -1, -1)
        for i in range(meta["n_pairs_ref"])
    ]

    print("Collecting tumor results...")
    tumor_results = collect_results(tumor_work, tumor_inputs,
                                     target_freqs_hz=array_geo.freqs_hz)
    print(f"  TD: {tumor_results['td_data'].shape}, "
          f"FD: {tumor_results['fd_data'].shape}")

    print("Collecting reference results...")
    ref_results = collect_results(ref_work, ref_inputs,
                                   target_freqs_hz=array_geo.freqs_hz)
    print(f"  TD: {ref_results['td_data'].shape}, "
          f"FD: {ref_results['fd_data'].shape}")

    # Calibrate
    fd_cal = calibrate(tumor_results["fd_data"], ref_results["fd_data"])
    print(f"Calibrated: max |S_cal| = {np.max(np.abs(fd_cal)):.2e}")

    # Add noise
    rng = np.random.default_rng(meta.get("seed", 42))
    noise_params = random_noise_params(rng=rng)
    fd_noisy = apply_noise_model(fd_cal, rng=rng, **noise_params)
    print(f"Noise: SNR={noise_params['snr_db']:.1f} dB, "
          f"phase={noise_params['phase_std_deg']:.1f} deg")

    # Export
    output_path = args.output or os.path.join(args.work_dir, "scan_3d.npz")
    np.savez_compressed(
        output_path,
        td_tumor=tumor_results["td_data"],
        td_ref=ref_results["td_data"],
        fd_cal=fd_cal,
        fd_noisy=fd_noisy,
        freqs_hz=tumor_results["freqs_hz"],
        dt=tumor_results["dt"],
        mode="3d",
        **{f"meta_{k}": v for k, v in meta.items()
           if not isinstance(v, (dict, list))},
    )
    print(f"\nSaved: {output_path}")
    print(f"  fd_noisy: {fd_noisy.shape}")


# --------------------------------------------------------------------------- #
# run-local: all phases on one node (for testing)
# --------------------------------------------------------------------------- #

def cmd_run_local(args):
    """Run all phases locally (useful for single-node GPU testing)."""
    # Phase 1
    cmd_prepare(args)

    # Phase 2 (local, no SLURM)
    from wavecare.synth.solver import run_simulation

    gpu = [args.gpu] if args.gpu is not None else None
    meta_path = os.path.join(args.work_dir, "scan_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    for label in ["tumor", "ref"]:
        work = os.path.join(args.work_dir, label)
        n = meta[f"n_pairs_{label}"]
        print(f"\nRunning {n} {label} simulations"
              f"{' (GPU ' + str(args.gpu) + ')' if gpu else ' (CPU)'}...")
        import time
        t0 = time.time()
        for i in range(n):
            infile = os.path.join(work, f"pair_{i:04d}.in")
            run_simulation(infile, gpu=gpu)
            if i == 0:
                elapsed = time.time() - t0
                print(f"  Pair 0: {elapsed:.1f}s "
                      f"(est. total: {elapsed*n/60:.1f} min)")
            elif (i+1) % max(1, n//6) == 0:
                total = time.time() - t0
                eta = total/(i+1) * (n-i-1)/60
                print(f"  [{i+1}/{n}] ETA: {eta:.1f} min")
        print(f"  {label}: {(time.time()-t0)/60:.1f} min")

    # Phase 3
    cmd_collect(args)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="SLURM-based 3D MWI scan generation")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- prepare --
    p_prep = sub.add_parser("prepare", help="Generate geometry + .in files")
    p_prep.add_argument("--phantom", required=True)
    p_prep.add_argument("--preset", default="umbmid",
                        choices=["umbmid", "maria", "mammowave"])
    p_prep.add_argument("--tumor-mm", type=float, default=12.0)
    p_prep.add_argument("--no-tumor", action="store_true")
    p_prep.add_argument("--seed", type=int, default=42)
    p_prep.add_argument("--dx", type=float, default=0.001,
                        help="Cell size in meters (default: 1mm)")
    p_prep.add_argument("--pad", type=int, default=40,
                        help="Padding cells (default: 40 = 4cm)")
    p_prep.add_argument("--work-dir", required=True)
    p_prep.add_argument("--data-dir", default=None)

    # -- submit --
    p_sub = sub.add_parser("submit", help="Submit SLURM array jobs")
    p_sub.add_argument("--work-dir", required=True)
    p_sub.add_argument("--partition", default="gpu")
    p_sub.add_argument("--gres", default="gpu:1")
    p_sub.add_argument("--account", default=None)
    p_sub.add_argument("--time-limit", default="00:30:00",
                       help="Per-task time limit (default: 30 min)")
    p_sub.add_argument("--dry-run", action="store_true",
                       help="Print sbatch commands without executing")

    # -- collect --
    p_col = sub.add_parser("collect", help="Gather results + calibrate + export")
    p_col.add_argument("--work-dir", required=True)
    p_col.add_argument("--output", default=None,
                       help="Output .npz path (default: work_dir/scan_3d.npz)")

    # -- run-local --
    p_loc = sub.add_parser("run-local", help="Run all phases locally (testing)")
    p_loc.add_argument("--phantom", required=True)
    p_loc.add_argument("--preset", default="umbmid",
                       choices=["umbmid", "maria", "mammowave"])
    p_loc.add_argument("--tumor-mm", type=float, default=12.0)
    p_loc.add_argument("--no-tumor", action="store_true")
    p_loc.add_argument("--seed", type=int, default=42)
    p_loc.add_argument("--dx", type=float, default=0.001)
    p_loc.add_argument("--pad", type=int, default=40)
    p_loc.add_argument("--work-dir", required=True)
    p_loc.add_argument("--data-dir", default=None)
    p_loc.add_argument("--output", default=None)
    p_loc.add_argument("--gpu", type=int, default=None,
                       help="GPU device ID (None = CPU)")

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "run-local":
        cmd_run_local(args)


if __name__ == "__main__":
    main()
