#!/usr/bin/env python3
"""Download and verify UWCEM breast phantoms.

The UWCEM Numerical Breast Phantom Repository provides MRI-derived
3D phantoms at 0.5mm resolution with 10 tissue types.

URL: https://uwcem.ece.wisc.edu/phantomRepository.html

Usage:
    python download_phantoms.py --data-dir /scratch/$USER/data/uwcem
    python download_phantoms.py --data-dir /scratch/$USER/data/uwcem --verify-only
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wavecare.synth.phantoms import load_uwcem_phantom, list_uwcem_phantoms

# Known UWCEM phantom IDs (from the repository)
KNOWN_PHANTOMS = [
    "010204", "012204", "012304", "012804", "013004",
    "020304", "040204", "050304", "062204", "070604", "071904",
]


def verify_phantoms(data_dir):
    """Load and verify all phantoms in data_dir."""
    phantom_ids = list_uwcem_phantoms(data_dir)
    if not phantom_ids:
        print(f"No phantoms found in {data_dir}")
        return []

    results = []
    print(f"\n{'ID':<10} {'Shape':<22} {'Voxel':<8} {'Class':<6} {'Status'}")
    print("-" * 60)

    for pid in phantom_ids:
        pdir = os.path.join(data_dir, pid)
        try:
            p = load_uwcem_phantom(pdir)
            shape_str = f"{p['shape'][0]}x{p['shape'][1]}x{p['shape'][2]}"
            cls = p['info'].get('classification', '?')
            print(f"{pid:<10} {shape_str:<22} {p['voxel_mm']:<8} {cls:<6} OK")
            results.append(pid)
        except Exception as e:
            print(f"{pid:<10} {'?':<22} {'?':<8} {'?':<6} FAIL: {e}")

    print(f"\n{len(results)}/{len(phantom_ids)} phantoms verified")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify UWCEM breast phantoms")
    parser.add_argument("--data-dir", required=True,
                        help="Directory to store phantoms")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing phantoms, don't download")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.verify_only:
        verify_phantoms(args.data_dir)
        return

    # Check which phantoms are already present
    existing = set(list_uwcem_phantoms(args.data_dir))
    missing = [p for p in KNOWN_PHANTOMS if p not in existing]

    if not missing:
        print("All known phantoms already present.")
        verify_phantoms(args.data_dir)
        return

    print("=" * 60)
    print("UWCEM Phantom Download")
    print("=" * 60)
    print(f"\nExisting: {len(existing)}/{len(KNOWN_PHANTOMS)}")
    print(f"Missing:  {missing}")
    print(f"\nThe UWCEM repository requires manual download:")
    print(f"  1. Go to: https://uwcem.ece.wisc.edu/phantomRepository.html")
    print(f"  2. Download each phantom's mtype.zip and breastInfo.txt")
    print(f"  3. Place them in: {args.data_dir}/<phantom_id>/")
    print(f"\nExpected structure:")
    print(f"  {args.data_dir}/")
    for pid in missing:
        print(f"    {pid}/")
        print(f"      breastInfo.txt")
        print(f"      mtype.zip")
    print(f"\nAfter downloading, run again with --verify-only to check.")

    # Verify whatever we have
    if existing:
        print(f"\nVerifying {len(existing)} existing phantoms:")
        verify_phantoms(args.data_dir)


if __name__ == "__main__":
    main()
