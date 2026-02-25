#!/usr/bin/env python3
"""SLURM-based 3D scan generation for DGX cluster.

Three-phase workflow:
  1. prepare  — load phantom, generate .in files (CPU, fast)
  2. submit   — sbatch array job, 1 GPU per simulation
  3. collect  — gather results, calibrate, add noise, export

Supports two modes:
  - positive: tumor scan + reference scan → calibrate → tumor signature
  - negative: ref_A + ref_B (different material perturbations) → residual only

Usage:
    # Positive scan (with tumor)
    python slurm_scan.py prepare \
        --phantom 071904 --preset umbmid --tumor-mm 12 \
        --work-dir /scratch/$USER/wavecare/scan_001

    # Negative scan (healthy, no tumor)
    python slurm_scan.py prepare \
        --phantom 071904 --preset umbmid --mode negative \
        --work-dir /scratch/$USER/wavecare/scan_002

    # Submit to SLURM
    python slurm_scan.py submit \
        --work-dir /scratch/$USER/wavecare/scan_001 \
        --partition gpu --gres gpu:1

    # Collect + calibrate + export
    python slurm_scan.py collect \
        --work-dir /scratch/$USER/wavecare/scan_001

    # Run all phases locally (testing):
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

# Default material perturbation for negative scans (5%)
_DEFAULT_PERTURBATION = 0.05


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
    if args.radius_cm is not None:
        array_geo.radius_m = args.radius_cm / 100.0
    mode = getattr(args, 'mode', 'positive')

    print("Loading phantom...")
    phantom = load_uwcem_phantom(os.path.join(data_dir, args.phantom))
    print(f"  Shape: {phantom['shape']}, voxel: {phantom['voxel_mm']} mm")
    print(f"  Mode: {mode}")
    print(f"  Radius: {array_geo.radius_m*100:.2f} cm")

    rng = np.random.default_rng(args.seed)

    if mode == "positive":
        meta = _prepare_positive(args, phantom, array_geo, rng)
    else:
        meta = _prepare_negative(args, phantom, array_geo, rng)

    # Save scan metadata
    meta_path = os.path.join(args.work_dir, "scan_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    total_sims = sum(v for k, v in meta.items() if k.startswith("n_pairs_"))
    print(f"\nPrepare complete. Metadata: {meta_path}")
    print(f"  Total sims: {total_sims}")


def _prepare_positive(args, phantom, array_geo, rng):
    """Prepare tumor + reference scans."""
    has_tumor = not args.no_tumor and args.tumor_mm > 0

    # --- Tumor scene ---
    tumor_work = os.path.join(args.work_dir, "tumor")
    os.makedirs(tumor_work, exist_ok=True)

    tumor_params = {"diameter_mm": args.tumor_mm, "shape": "sphere"} if has_tumor else None
    tumor_scene = generate_scene(
        phantom, has_tumor=has_tumor, rng=rng, tumor_params=tumor_params)

    print("Preparing 3D geometry (tumor)...")
    tumor_geo = prepare_3d_geometry(tumor_scene, dx=args.dx, pad_cells=args.pad)
    write_geometry_files(tumor_geo, tumor_work)
    tumor_inputs = generate_gprmax_inputs_3d(
        tumor_geo,
        array_geo,
        tumor_work,
        min_clearance_m=args.min_clearance_mm / 1000.0,
    )
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
    ref_inputs = generate_gprmax_inputs_3d(
        ref_geo,
        array_geo,
        ref_work,
        min_clearance_m=args.min_clearance_mm / 1000.0,
    )
    print(f"  {len(ref_inputs)} .in files written")

    # Metadata
    meta = {
        "mode": "positive",
        "phantom_id": args.phantom,
        "preset": args.preset,
        "has_tumor": has_tumor,
        "tumor_mm": args.tumor_mm if has_tumor else 0,
        "seed": args.seed,
        "dx": args.dx,
        "ant_radius_cm": array_geo.radius_m * 100.0,
        "min_clearance_mm": args.min_clearance_mm,
        "pad_cells": args.pad,
        "n_pairs_tumor": len(tumor_inputs),
        "n_pairs_ref": len(ref_inputs),
        "geo_shape": list(tumor_geo["geo_shape"]),
        "domain_m": list(tumor_geo["domain_m"]),
        "n_cells": int(np.prod(tumor_geo["geo_shape"])),
        "dirs": ["tumor", "ref"],
    }
    if tumor_scene.get("tumor_info"):
        ti = tumor_scene["tumor_info"]
        for k, v in ti.items():
            meta[f"tumor_{k}"] = v if not isinstance(v, np.ndarray) else v.tolist()

    meta["scene_metadata"] = scene_to_metadata(
        tumor_scene, scan_id=0, phant_id=args.phantom)

    return meta


def _prepare_negative(args, phantom, array_geo, rng):
    """Prepare two reference scans with different material perturbations."""
    perturbation = getattr(args, 'perturbation', _DEFAULT_PERTURBATION)

    # --- Reference A ---
    ref_a_work = os.path.join(args.work_dir, "ref_a")
    os.makedirs(ref_a_work, exist_ok=True)

    rng_a = np.random.default_rng(args.seed)
    scene_a = generate_scene(phantom, has_tumor=False, rng=rng_a)

    print("Preparing 3D geometry (ref_A)...")
    geo_a = prepare_3d_geometry(scene_a, dx=args.dx, pad_cells=args.pad)
    mat_rng_a = np.random.default_rng(args.seed + 2_000_000)
    write_geometry_files(geo_a, ref_a_work,
                         perturbation=perturbation, rng=mat_rng_a)
    inputs_a = generate_gprmax_inputs_3d(
        geo_a,
        array_geo,
        ref_a_work,
        min_clearance_m=args.min_clearance_mm / 1000.0,
    )
    print(f"  Domain: {geo_a['geo_shape']} = "
          f"{np.prod(geo_a['geo_shape'])/1e6:.1f}M cells")
    print(f"  {len(inputs_a)} .in files written")

    # --- Reference B (different material perturbation) ---
    ref_b_work = os.path.join(args.work_dir, "ref_b")
    os.makedirs(ref_b_work, exist_ok=True)

    rng_b = np.random.default_rng(args.seed + 1_000_000)
    scene_b = generate_scene(phantom, has_tumor=False, rng=rng_b)

    print("Preparing 3D geometry (ref_B)...")
    geo_b = prepare_3d_geometry(scene_b, dx=args.dx, pad_cells=args.pad)
    mat_rng_b = np.random.default_rng(args.seed + 3_000_000)
    write_geometry_files(geo_b, ref_b_work,
                         perturbation=perturbation, rng=mat_rng_b)
    inputs_b = generate_gprmax_inputs_3d(
        geo_b,
        array_geo,
        ref_b_work,
        min_clearance_m=args.min_clearance_mm / 1000.0,
    )
    print(f"  {len(inputs_b)} .in files written")

    # Metadata
    meta = {
        "mode": "negative",
        "phantom_id": args.phantom,
        "preset": args.preset,
        "has_tumor": False,
        "tumor_mm": 0,
        "seed": args.seed,
        "dx": args.dx,
        "ant_radius_cm": array_geo.radius_m * 100.0,
        "min_clearance_mm": args.min_clearance_mm,
        "pad_cells": args.pad,
        "perturbation": perturbation,
        "n_pairs_ref_a": len(inputs_a),
        "n_pairs_ref_b": len(inputs_b),
        "geo_shape": list(geo_a["geo_shape"]),
        "domain_m": list(geo_a["domain_m"]),
        "n_cells": int(np.prod(geo_a["geo_shape"])),
        "dirs": ["ref_a", "ref_b"],
    }
    meta["scene_metadata"] = scene_to_metadata(
        scene_a, scan_id=0, phant_id=args.phantom)

    return meta


# --------------------------------------------------------------------------- #
# Phase 2: submit SLURM
# --------------------------------------------------------------------------- #

def _submit_array(work_dir, label, n_pairs, slurm_script, args, phantom_id):
    """Submit one SLURM array job. Returns job ID or None."""
    sub_dir = os.path.join(work_dir, label)
    cmd = [
        "sbatch",
        f"--array=0-{n_pairs-1}",
        f"--partition={args.partition}",
        f"--gres={args.gres}",
        f"--job-name=wc_{label}_{phantom_id}",
        f"--output={sub_dir}/slurm_%A_%a.log",
        f"--time={args.time_limit}",
        slurm_script,
        sub_dir,
    ]
    if args.account:
        cmd.insert(1, f"--account={args.account}")

    print(f"  Submitting {label} ({n_pairs} tasks)...")
    print(f"    {' '.join(cmd)}")

    if args.dry_run:
        return None

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"    {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()}", file=sys.stderr)
        return None

    # Parse job ID from "Submitted batch job 12345"
    parts = result.stdout.strip().split()
    return parts[-1] if parts else None


def cmd_submit(args):
    """Submit SLURM array jobs."""
    meta_path = os.path.join(args.work_dir, "scan_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    slurm_script = os.path.join(_SCRIPT_DIR, "slurm_sim.sh")
    phantom_id = meta["phantom_id"]
    mode = meta.get("mode", "positive")
    job_ids = []

    if mode == "positive":
        jid = _submit_array(args.work_dir, "tumor", meta["n_pairs_tumor"],
                            slurm_script, args, phantom_id)
        if jid:
            job_ids.append(jid)
        jid = _submit_array(args.work_dir, "ref", meta["n_pairs_ref"],
                            slurm_script, args, phantom_id)
        if jid:
            job_ids.append(jid)
    else:
        jid = _submit_array(args.work_dir, "ref_a", meta["n_pairs_ref_a"],
                            slurm_script, args, phantom_id)
        if jid:
            job_ids.append(jid)
        jid = _submit_array(args.work_dir, "ref_b", meta["n_pairs_ref_b"],
                            slurm_script, args, phantom_id)
        if jid:
            job_ids.append(jid)

    # Save job IDs for dependency tracking
    if job_ids:
        jid_path = os.path.join(args.work_dir, "slurm_job_ids.json")
        with open(jid_path, 'w') as f:
            json.dump(job_ids, f)
        print(f"\nJob IDs: {job_ids}")


# --------------------------------------------------------------------------- #
# Phase 3: collect
# --------------------------------------------------------------------------- #

def cmd_collect(args):
    """Gather simulation outputs, calibrate, add noise, export."""
    meta_path = os.path.join(args.work_dir, "scan_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    array_geo = _get_preset(meta["preset"])
    mode = meta.get("mode", "positive")

    if mode == "positive":
        _collect_positive(args, meta, array_geo)
    else:
        _collect_negative(args, meta, array_geo)


def _collect_positive(args, meta, array_geo):
    """Collect tumor + reference, calibrate, export."""
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

    # Calibrate: tumor - ref
    fd_cal = calibrate(tumor_results["fd_data"], ref_results["fd_data"])
    print(f"Calibrated: max |S_cal| = {np.max(np.abs(fd_cal)):.2e}")

    _export_scan(args, meta, fd_cal, tumor_results, ref_results, label=1)


def _collect_negative(args, meta, array_geo):
    """Collect two reference scans, calibrate, export."""
    ref_a_work = os.path.join(args.work_dir, "ref_a")
    ref_b_work = os.path.join(args.work_dir, "ref_b")

    inputs_a = [
        (os.path.join(ref_a_work, f"pair_{i:04d}.in"), -1, -1)
        for i in range(meta["n_pairs_ref_a"])
    ]
    inputs_b = [
        (os.path.join(ref_b_work, f"pair_{i:04d}.in"), -1, -1)
        for i in range(meta["n_pairs_ref_b"])
    ]

    print("Collecting ref_A results...")
    results_a = collect_results(ref_a_work, inputs_a,
                                 target_freqs_hz=array_geo.freqs_hz)
    print(f"  TD: {results_a['td_data'].shape}, "
          f"FD: {results_a['fd_data'].shape}")

    print("Collecting ref_B results...")
    results_b = collect_results(ref_b_work, inputs_b,
                                 target_freqs_hz=array_geo.freqs_hz)

    # Calibrate: ref_A - ref_B → residual from perturbation only
    fd_cal = calibrate(results_a["fd_data"], results_b["fd_data"])
    print(f"Calibrated (negative): max |S_cal| = {np.max(np.abs(fd_cal)):.2e}")

    _export_scan(args, meta, fd_cal, results_a, results_b, label=0)


def _export_scan(args, meta, fd_cal, results_scan, results_ref, label):
    """Add noise and export scan."""
    rng = np.random.default_rng(meta.get("seed", 42))
    noise_params = random_noise_params(rng=rng)
    fd_noisy = apply_noise_model(fd_cal, rng=rng, **noise_params)
    print(f"Noise: SNR={noise_params['snr_db']:.1f} dB, "
          f"phase={noise_params['phase_std_deg']:.1f} deg")

    output_path = args.output or os.path.join(args.work_dir, "scan_3d.npz")
    np.savez_compressed(
        output_path,
        td_scan=results_scan["td_data"],
        td_ref=results_ref["td_data"],
        fd_cal=fd_cal,
        fd_noisy=fd_noisy,
        freqs_hz=results_scan["freqs_hz"],
        dt=results_scan["dt"],
        label=label,
        mode="3d",
        **{f"meta_{k}": v for k, v in meta.items()
           if not isinstance(v, (dict, list))},
    )
    print(f"\nSaved: {output_path}")
    print(f"  fd_noisy: {fd_noisy.shape}, label={label}")


# --------------------------------------------------------------------------- #
# run-local: all phases on one node (for testing)
# --------------------------------------------------------------------------- #

def cmd_run_local(args):
    """Run all phases locally (useful for single-node GPU testing)."""
    # Phase 1
    cmd_prepare(args)

    # Phase 2 (local, no SLURM)
    from wavecare.synth.solver import run_simulation
    import time

    gpu = [args.gpu] if args.gpu is not None else None
    meta_path = os.path.join(args.work_dir, "scan_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    dirs = meta.get("dirs", ["tumor", "ref"])
    for label in dirs:
        work = os.path.join(args.work_dir, label)
        # Find the right n_pairs key
        n_key = f"n_pairs_{label}"
        n = meta[n_key]
        print(f"\nRunning {n} {label} simulations"
              f"{' (GPU ' + str(args.gpu) + ')' if gpu else ' (CPU)'}...")
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

def _add_common_prepare_args(p):
    """Add arguments shared by prepare and run-local subcommands."""
    p.add_argument("--phantom", required=True)
    p.add_argument("--preset", default="umbmid",
                   choices=["umbmid", "maria", "mammowave"])
    p.add_argument("--mode", default="positive",
                   choices=["positive", "negative"],
                   help="positive=tumor+ref, negative=ref_A+ref_B")
    p.add_argument("--tumor-mm", type=float, default=12.0)
    p.add_argument("--no-tumor", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dx", type=float, default=0.001,
                   help="Cell size in meters (default: 1mm)")
    p.add_argument("--pad", type=int, default=50,
                   help="Padding cells (default: 50 = 5cm)")
    p.add_argument("--radius-cm", type=float, default=None,
                   help="Override antenna ring radius (cm)")
    p.add_argument("--min-clearance-mm", type=float, default=0.0,
                   help="Required antenna-to-tissue clearance (mm)")
    p.add_argument("--perturbation", type=float, default=_DEFAULT_PERTURBATION,
                   help="Material perturbation for negative scans (default: 0.05)")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--data-dir", default=None)


def main():
    parser = argparse.ArgumentParser(
        description="SLURM-based 3D MWI scan generation")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- prepare --
    p_prep = sub.add_parser("prepare", help="Generate geometry + .in files")
    _add_common_prepare_args(p_prep)

    # -- submit --
    p_sub = sub.add_parser("submit", help="Submit SLURM array jobs")
    p_sub.add_argument("--work-dir", required=True)
    p_sub.add_argument("--partition", default="gpu")
    p_sub.add_argument("--gres", default="gpu:1")
    p_sub.add_argument("--account", default=None)
    p_sub.add_argument("--time-limit", default="01:00:00",
                       help="Per-task time limit (default: 1 hour)")
    p_sub.add_argument("--dry-run", action="store_true",
                       help="Print sbatch commands without executing")

    # -- collect --
    p_col = sub.add_parser("collect", help="Gather results + calibrate + export")
    p_col.add_argument("--work-dir", required=True)
    p_col.add_argument("--output", default=None,
                       help="Output .npz path (default: work_dir/scan_3d.npz)")

    # -- run-local --
    p_loc = sub.add_parser("run-local", help="Run all phases locally (testing)")
    _add_common_prepare_args(p_loc)
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
