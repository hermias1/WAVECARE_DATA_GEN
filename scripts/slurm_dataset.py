#!/usr/bin/env python3
"""Generate a multi-scan 3D dataset via SLURM.

Orchestrates slurm_scan.py for multiple scans with varying
tumor sizes and seeds.

Usage:
    # Prepare all scans (run on login node)
    python slurm_dataset.py prepare \
        --phantom 071904 --preset umbmid --n-scans 50 \
        --tumor-range 5 25 --base-dir /scratch/$USER/wavecare/dataset_001

    # Submit all to SLURM
    python slurm_dataset.py submit \
        --base-dir /scratch/$USER/wavecare/dataset_001 \
        --partition gpu

    # Check progress
    python slurm_dataset.py status --base-dir /scratch/$USER/wavecare/dataset_001

    # Collect all into one dataset
    python slurm_dataset.py collect \
        --base-dir /scratch/$USER/wavecare/dataset_001 \
        --output /results/dataset_3d.npz
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)


def cmd_prepare(args):
    """Prepare all scans (geometry + .in files)."""
    rng = np.random.default_rng(args.seed)

    if args.tumor_range:
        tumor_sizes = rng.uniform(args.tumor_range[0], args.tumor_range[1],
                                   args.n_scans)
    else:
        tumor_sizes = np.full(args.n_scans, args.tumor_mm)

    scan_seeds = rng.integers(0, 2**31, size=args.n_scans)

    os.makedirs(args.base_dir, exist_ok=True)

    # Save dataset plan
    plan = {
        "phantom_id": args.phantom,
        "preset": args.preset,
        "n_scans": args.n_scans,
        "dx": args.dx,
        "seed": args.seed,
        "scans": [],
    }

    for i in range(args.n_scans):
        scan_dir = os.path.join(args.base_dir, f"scan_{i:04d}")
        tumor_mm = float(tumor_sizes[i])
        seed = int(scan_seeds[i])

        print(f"\n--- Preparing scan {i}/{args.n_scans}: "
              f"tumor={tumor_mm:.1f}mm, seed={seed} ---")

        cmd = [
            sys.executable, os.path.join(_SCRIPT_DIR, "slurm_scan.py"),
            "prepare",
            "--phantom", args.phantom,
            "--preset", args.preset,
            "--tumor-mm", str(tumor_mm),
            "--seed", str(seed),
            "--dx", str(args.dx),
            "--work-dir", scan_dir,
        ]
        if args.data_dir:
            cmd += ["--data-dir", args.data_dir]

        subprocess.run(cmd, check=True)

        plan["scans"].append({
            "id": i,
            "dir": scan_dir,
            "tumor_mm": tumor_mm,
            "seed": seed,
        })

    plan_path = os.path.join(args.base_dir, "dataset_plan.json")
    with open(plan_path, 'w') as f:
        json.dump(plan, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Prepared {args.n_scans} scans")
    print(f"Plan: {plan_path}")
    print(f"Next: python slurm_dataset.py submit --base-dir {args.base_dir}")


def cmd_submit(args):
    """Submit all scans to SLURM."""
    plan_path = os.path.join(args.base_dir, "dataset_plan.json")
    with open(plan_path) as f:
        plan = json.load(f)

    for scan in plan["scans"]:
        print(f"\nSubmitting scan {scan['id']}...")
        cmd = [
            sys.executable, os.path.join(_SCRIPT_DIR, "slurm_scan.py"),
            "submit",
            "--work-dir", scan["dir"],
            "--partition", args.partition,
            "--gres", args.gres,
            "--time-limit", args.time_limit,
        ]
        if args.account:
            cmd += ["--account", args.account]
        if args.dry_run:
            cmd += ["--dry-run"]

        subprocess.run(cmd, check=True)

    print(f"\nAll {len(plan['scans'])} scans submitted.")
    print(f"Monitor: squeue -u $USER")
    print(f"Next: python slurm_dataset.py status --base-dir {args.base_dir}")


def cmd_status(args):
    """Check how many simulations have completed."""
    plan_path = os.path.join(args.base_dir, "dataset_plan.json")
    with open(plan_path) as f:
        plan = json.load(f)

    total_done = 0
    total_expected = 0

    for scan in plan["scans"]:
        meta_path = os.path.join(scan["dir"], "scan_meta.json")
        if not os.path.exists(meta_path):
            print(f"  scan_{scan['id']:04d}: NOT PREPARED")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        n_expected = meta["n_pairs_tumor"] + meta["n_pairs_ref"]
        total_expected += n_expected

        # Count .out files
        n_done = 0
        for label in ["tumor", "ref"]:
            work = os.path.join(scan["dir"], label)
            n_pairs = meta[f"n_pairs_{label}"]
            for i in range(n_pairs):
                if os.path.exists(os.path.join(work, f"pair_{i:04d}.out")):
                    n_done += 1

        total_done += n_done
        status = "DONE" if n_done == n_expected else f"{n_done}/{n_expected}"
        print(f"  scan_{scan['id']:04d}: {status} "
              f"(tumor={scan['tumor_mm']:.1f}mm)")

    pct = 100 * total_done / total_expected if total_expected > 0 else 0
    print(f"\nTotal: {total_done}/{total_expected} ({pct:.0f}%)")


def cmd_collect(args):
    """Collect all scans into a single dataset."""
    plan_path = os.path.join(args.base_dir, "dataset_plan.json")
    with open(plan_path) as f:
        plan = json.load(f)

    all_fd_noisy = []
    all_fd_cal = []
    all_labels = []

    for scan in plan["scans"]:
        print(f"Collecting scan {scan['id']}...")
        cmd = [
            sys.executable, os.path.join(_SCRIPT_DIR, "slurm_scan.py"),
            "collect",
            "--work-dir", scan["dir"],
        ]
        subprocess.run(cmd, check=True)

        # Load the per-scan result
        scan_npz = os.path.join(scan["dir"], "scan_3d.npz")
        d = np.load(scan_npz)
        all_fd_noisy.append(d["fd_noisy"])
        all_fd_cal.append(d["fd_cal"])
        all_labels.append(1)  # has tumor

    fd_dataset = np.stack(all_fd_noisy)
    fd_cal_dataset = np.stack(all_fd_cal)
    labels = np.array(all_labels)

    output_path = args.output or os.path.join(args.base_dir, "dataset_3d.npz")
    np.savez_compressed(
        output_path,
        fd_data=fd_dataset,
        fd_cal=fd_cal_dataset,
        labels=labels,
        freqs_hz=np.load(os.path.join(
            plan["scans"][0]["dir"], "scan_3d.npz"))["freqs_hz"],
        n_scans=len(plan["scans"]),
        phantom_id=plan["phantom_id"],
        preset=plan["preset"],
        mode="3d",
    )

    print(f"\n{'='*60}")
    print(f"Dataset: {fd_dataset.shape} → {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-scan 3D dataset generation via SLURM")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- prepare --
    p = sub.add_parser("prepare")
    p.add_argument("--phantom", required=True)
    p.add_argument("--preset", default="umbmid",
                   choices=["umbmid", "maria", "mammowave"])
    p.add_argument("--n-scans", type=int, required=True)
    p.add_argument("--tumor-mm", type=float, default=12.0)
    p.add_argument("--tumor-range", type=float, nargs=2, default=None,
                   metavar=("MIN", "MAX"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dx", type=float, default=0.001)
    p.add_argument("--base-dir", required=True)
    p.add_argument("--data-dir", default=None)

    # -- submit --
    p = sub.add_parser("submit")
    p.add_argument("--base-dir", required=True)
    p.add_argument("--partition", default="gpu")
    p.add_argument("--gres", default="gpu:1")
    p.add_argument("--account", default=None)
    p.add_argument("--time-limit", default="00:30:00")
    p.add_argument("--dry-run", action="store_true")

    # -- status --
    p = sub.add_parser("status")
    p.add_argument("--base-dir", required=True)

    # -- collect --
    p = sub.add_parser("collect")
    p.add_argument("--base-dir", required=True)
    p.add_argument("--output", default=None)

    args = parser.parse_args()
    {"prepare": cmd_prepare, "submit": cmd_submit,
     "status": cmd_status, "collect": cmd_collect}[args.command](args)


if __name__ == "__main__":
    main()
