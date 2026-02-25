#!/usr/bin/env python3
"""Generate a multi-scan 3D dataset via SLURM.

Orchestrates slurm_scan.py for multiple scans with varying
tumor sizes, multiple phantoms, and positive/negative labels.

Usage:
    # Prepare all scans (run on login node)
    python slurm_dataset.py prepare \
        --phantom-dir /scratch/$USER/data/uwcem \
        --preset umbmid \
        --n-positive 35 --n-negative 15 \
        --tumor-range 5 25 \
        --base-dir /scratch/$USER/wavecare/dataset_001

    # Submit all to SLURM
    python slurm_dataset.py submit \
        --base-dir /scratch/$USER/wavecare/dataset_001 \
        --partition gpu

    # Check progress
    python slurm_dataset.py status --base-dir /scratch/$USER/wavecare/dataset_001

    # Re-submit failed simulations
    python slurm_dataset.py resubmit --base-dir /scratch/$USER/wavecare/dataset_001

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

from wavecare.synth.phantoms import list_uwcem_phantoms


def cmd_prepare(args):
    """Prepare all scans (geometry + .in files)."""
    rng = np.random.default_rng(args.seed)

    # Discover available phantoms
    phantom_ids = list_uwcem_phantoms(args.phantom_dir)
    if not phantom_ids:
        print(f"ERROR: No phantoms found in {args.phantom_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(phantom_ids)} phantoms: {phantom_ids}")

    n_total = args.n_positive + args.n_negative

    # Assign tumors sizes for positive scans
    if args.tumor_range:
        tumor_sizes = rng.uniform(args.tumor_range[0], args.tumor_range[1],
                                   args.n_positive)
    else:
        tumor_sizes = np.full(args.n_positive, args.tumor_mm)

    scan_seeds = rng.integers(0, 2**31, size=n_total)

    # Build scan plan: positive first, then negative, distributed round-robin
    scan_plan = []
    for i in range(args.n_positive):
        scan_plan.append({
            "id": i,
            "phantom_id": phantom_ids[i % len(phantom_ids)],
            "mode": "positive",
            "tumor_mm": float(tumor_sizes[i]),
            "seed": int(scan_seeds[i]),
        })
    for i in range(args.n_negative):
        idx = args.n_positive + i
        scan_plan.append({
            "id": idx,
            "phantom_id": phantom_ids[idx % len(phantom_ids)],
            "mode": "negative",
            "tumor_mm": 0,
            "seed": int(scan_seeds[idx]),
        })

    # Shuffle so positive/negative are interleaved in SLURM queue
    rng.shuffle(scan_plan)

    os.makedirs(args.base_dir, exist_ok=True)

    # Save dataset plan
    plan = {
        "phantom_dir": args.phantom_dir,
        "phantom_ids": phantom_ids,
        "preset": args.preset,
        "n_positive": args.n_positive,
        "n_negative": args.n_negative,
        "n_total": n_total,
        "dx": args.dx,
        "pad": args.pad,
        "radius_cm": args.radius_cm,
        "min_clearance_mm": args.min_clearance_mm,
        "center_mode": args.center_mode,
        "center_search_mm": args.center_search_mm,
        "seed": args.seed,
        "scans": [],
    }

    for scan in scan_plan:
        scan_dir = os.path.join(args.base_dir, f"scan_{scan['id']:04d}")

        print(f"\n--- Preparing scan {scan['id']}/{n_total}: "
              f"mode={scan['mode']}, phantom={scan['phantom_id']}, "
              f"tumor={scan['tumor_mm']:.1f}mm, seed={scan['seed']} ---")

        cmd = [
            sys.executable, os.path.join(_SCRIPT_DIR, "slurm_scan.py"),
            "prepare",
            "--phantom", scan["phantom_id"],
            "--preset", args.preset,
            "--mode", scan["mode"],
            "--tumor-mm", str(scan["tumor_mm"]),
            "--seed", str(scan["seed"]),
            "--dx", str(args.dx),
            "--pad", str(args.pad),
            "--center-mode", args.center_mode,
            "--center-search-mm", str(args.center_search_mm),
            "--min-clearance-mm", str(args.min_clearance_mm),
            "--work-dir", scan_dir,
            "--data-dir", args.phantom_dir,
        ]
        if args.radius_cm is not None:
            cmd += ["--radius-cm", str(args.radius_cm)]
        subprocess.run(cmd, check=True)

        plan["scans"].append({
            "id": scan["id"],
            "dir": scan_dir,
            "phantom_id": scan["phantom_id"],
            "mode": scan["mode"],
            "tumor_mm": scan["tumor_mm"],
            "seed": scan["seed"],
        })

    plan_path = os.path.join(args.base_dir, "dataset_plan.json")
    with open(plan_path, 'w') as f:
        json.dump(plan, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Prepared {n_total} scans ({args.n_positive} positive, "
          f"{args.n_negative} negative)")
    print(f"Phantoms used: {len(phantom_ids)}")
    print(f"Plan: {plan_path}")
    print(f"Next: python slurm_dataset.py submit --base-dir {args.base_dir}")


def cmd_submit(args):
    """Submit all scans to SLURM."""
    plan_path = os.path.join(args.base_dir, "dataset_plan.json")
    with open(plan_path) as f:
        plan = json.load(f)

    for scan in plan["scans"]:
        print(f"\nSubmitting scan {scan['id']} "
              f"({scan['mode']}, {scan['phantom_id']})...")
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
    total_missing = 0

    for scan in plan["scans"]:
        meta_path = os.path.join(scan["dir"], "scan_meta.json")
        if not os.path.exists(meta_path):
            print(f"  scan_{scan['id']:04d}: NOT PREPARED")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        dirs = meta.get("dirs", ["tumor", "ref"])
        n_expected = 0
        n_done = 0
        missing_pairs = {}

        for label in dirs:
            work = os.path.join(scan["dir"], label)
            n_key = f"n_pairs_{label}"
            n_pairs = meta[n_key]
            n_expected += n_pairs
            label_missing = []
            for i in range(n_pairs):
                if os.path.exists(os.path.join(work, f"pair_{i:04d}.out")):
                    n_done += 1
                else:
                    label_missing.append(i)
            if label_missing:
                missing_pairs[label] = label_missing

        total_done += n_done
        total_expected += n_expected
        total_missing += (n_expected - n_done)

        status = "DONE" if n_done == n_expected else f"{n_done}/{n_expected}"
        mode_tag = "+" if scan["mode"] == "positive" else "-"
        extra = f" tumor={scan['tumor_mm']:.0f}mm" if scan["mode"] == "positive" else ""
        print(f"  scan_{scan['id']:04d} [{mode_tag}] {scan['phantom_id']}: "
              f"{status}{extra}")

    pct = 100 * total_done / total_expected if total_expected > 0 else 0
    print(f"\nTotal: {total_done}/{total_expected} ({pct:.0f}%)")
    if total_missing > 0:
        print(f"Missing: {total_missing} simulations")
        print(f"Use 'resubmit' to re-run failed simulations.")


def cmd_resubmit(args):
    """Re-submit only failed/missing simulations."""
    plan_path = os.path.join(args.base_dir, "dataset_plan.json")
    with open(plan_path) as f:
        plan = json.load(f)

    slurm_script = os.path.join(_SCRIPT_DIR, "slurm_sim.sh")
    n_resubmitted = 0

    for scan in plan["scans"]:
        meta_path = os.path.join(scan["dir"], "scan_meta.json")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        dirs = meta.get("dirs", ["tumor", "ref"])
        for label in dirs:
            work = os.path.join(scan["dir"], label)
            n_key = f"n_pairs_{label}"
            n_pairs = meta[n_key]

            missing = []
            for i in range(n_pairs):
                if not os.path.exists(os.path.join(work, f"pair_{i:04d}.out")):
                    missing.append(i)

            if not missing:
                continue

            array_spec = ",".join(str(i) for i in missing)
            cmd = [
                "sbatch",
                f"--array={array_spec}",
                f"--partition={args.partition}",
                f"--gres={args.gres}",
                f"--job-name=wc_re_{label}_{scan['phantom_id']}",
                f"--output={work}/slurm_%A_%a.log",
                f"--time={args.time_limit}",
                slurm_script,
                work,
            ]
            if args.account:
                cmd.insert(1, f"--account={args.account}")

            print(f"  scan_{scan['id']:04d}/{label}: "
                  f"resubmitting {len(missing)} missing sims")

            if not args.dry_run:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"    {result.stdout.strip()}")
                else:
                    print(f"    ERROR: {result.stderr.strip()}")
            n_resubmitted += len(missing)

    print(f"\nResubmitted {n_resubmitted} simulations.")


def cmd_collect(args):
    """Collect all scans into a single dataset."""
    plan_path = os.path.join(args.base_dir, "dataset_plan.json")
    with open(plan_path) as f:
        plan = json.load(f)

    all_fd_noisy = []
    all_fd_cal = []
    all_labels = []

    for scan in plan["scans"]:
        print(f"Collecting scan {scan['id']} "
              f"({scan['mode']}, {scan['phantom_id']})...")

        cmd = [
            sys.executable, os.path.join(_SCRIPT_DIR, "slurm_scan.py"),
            "collect",
            "--work-dir", scan["dir"],
        ]
        subprocess.run(cmd, check=True)

        # Load the per-scan result
        scan_npz = os.path.join(scan["dir"], "scan_3d.npz")
        d = np.load(scan_npz)  # nosec — our own generated files
        all_fd_noisy.append(d["fd_noisy"])
        all_fd_cal.append(d["fd_cal"])

        # Label from scan mode
        label = 1 if scan["mode"] == "positive" else 0
        all_labels.append(label)

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
            plan["scans"][0]["dir"], "scan_3d.npz"))["freqs_hz"],  # nosec
        n_scans=len(plan["scans"]),
        n_positive=int(labels.sum()),
        n_negative=int((1 - labels).sum()),
        phantom_ids=[s["phantom_id"] for s in plan["scans"]],
        preset=plan["preset"],
        mode="3d",
    )

    print(f"\n{'='*60}")
    print(f"Dataset: {fd_dataset.shape} -> {output_path}")
    print(f"Labels: {int(labels.sum())} positive, "
          f"{int((1-labels).sum())} negative")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-scan 3D dataset generation via SLURM")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- prepare --
    p = sub.add_parser("prepare")
    p.add_argument("--phantom-dir", required=True,
                   help="Directory containing UWCEM phantom subdirectories")
    p.add_argument("--preset", default="umbmid",
                   choices=["umbmid", "maria", "mammowave"])
    p.add_argument("--n-positive", type=int, default=35,
                   help="Number of scans with tumor (default: 35)")
    p.add_argument("--n-negative", type=int, default=15,
                   help="Number of healthy scans (default: 15)")
    p.add_argument("--tumor-mm", type=float, default=12.0,
                   help="Fixed tumor diameter if no --tumor-range")
    p.add_argument("--tumor-range", type=float, nargs=2, default=None,
                   metavar=("MIN", "MAX"),
                   help="Tumor diameter range in mm (e.g., 5 25)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dx", type=float, default=0.001)
    p.add_argument("--pad", type=int, default=50)
    p.add_argument("--radius-cm", type=float, default=None,
                   help="Override antenna ring radius (cm)")
    p.add_argument("--min-clearance-mm", type=float, default=0.0,
                   help="Required antenna-to-tissue clearance (mm)")
    p.add_argument("--center-mode", default="auto",
                   choices=["auto", "volume", "tissue_centroid",
                            "skin_centroid", "ring_fit"])
    p.add_argument("--center-search-mm", type=float, default=20.0)
    p.add_argument("--base-dir", required=True)

    # -- submit --
    p = sub.add_parser("submit")
    p.add_argument("--base-dir", required=True)
    p.add_argument("--partition", default="gpu")
    p.add_argument("--gres", default="gpu:1")
    p.add_argument("--account", default=None)
    p.add_argument("--time-limit", default="01:00:00")
    p.add_argument("--dry-run", action="store_true")

    # -- status --
    p = sub.add_parser("status")
    p.add_argument("--base-dir", required=True)

    # -- resubmit --
    p = sub.add_parser("resubmit")
    p.add_argument("--base-dir", required=True)
    p.add_argument("--partition", default="gpu")
    p.add_argument("--gres", default="gpu:1")
    p.add_argument("--account", default=None)
    p.add_argument("--time-limit", default="01:00:00")
    p.add_argument("--dry-run", action="store_true")

    # -- collect --
    p = sub.add_parser("collect")
    p.add_argument("--base-dir", required=True)
    p.add_argument("--output", default=None)

    args = parser.parse_args()
    {
        "prepare": cmd_prepare,
        "submit": cmd_submit,
        "status": cmd_status,
        "resubmit": cmd_resubmit,
        "collect": cmd_collect,
    }[args.command](args)


if __name__ == "__main__":
    main()
