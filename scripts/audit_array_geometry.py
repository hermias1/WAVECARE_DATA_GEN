#!/usr/bin/env python3
"""Audit minimal (radius, pad) satisfying antenna placement constraints.

For each phantom, this script searches over ring radius and padding values and
reports the first configuration that is physically valid:
- no antenna outside domain,
- no antenna in tissue,
- minimum antenna-to-tissue clearance satisfied.
"""

import argparse
import json
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)

from wavecare.acqui import presets
from wavecare.synth.phantoms import list_uwcem_phantoms, load_uwcem_phantom
from wavecare.synth.solver import prepare_3d_geometry, evaluate_array_placement


def _get_preset(name):
    return {
        "umbmid": presets.umbmid_gen2,
        "maria": presets.maria_m5,
        "mammowave": presets.mammowave,
    }[name]()


def _radius_grid(min_cm, max_cm, step_cm):
    n = int(np.floor((max_cm - min_cm) / step_cm)) + 1
    return [round(min_cm + i * step_cm, 6) for i in range(max(0, n))]


def _parse_pads(pads_text):
    vals = []
    for tok in pads_text.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    if not vals:
        raise ValueError("No valid pad values parsed from --pads")
    return sorted(set(vals))


def _pick_best(candidates):
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda x: (
            x["radius_cm"],
            x["pad_cells"],
            abs(x["center_shift_x_mm"]) + abs(x["center_shift_y_mm"]),
        ),
    )


def _pick_closest_failure(rows):
    if not rows:
        return None
    return min(
        rows,
        key=lambda x: (
            x["n_oob"],
            x["n_in_tissue"],
            max(0.0, x["required_clearance_mm"] - x["min_clearance_mm"]),
            x["radius_cm"],
            x["pad_cells"],
        ),
    )


def audit_phantom(
    phantom_id,
    phantom_dir,
    preset_name,
    dx,
    pads,
    radii_cm,
    min_clearance_mm,
    center_mode,
    center_search_mm,
):
    phantom = load_uwcem_phantom(os.path.join(phantom_dir, phantom_id))
    scene = {"mtype": phantom["mtype"], "voxel_mm": phantom["voxel_mm"]}

    valid_rows = []
    all_rows = []
    for pad in pads:
        geo = prepare_3d_geometry(scene, dx=dx, pad_cells=pad)
        for radius_cm in radii_cm:
            array_geo = _get_preset(preset_name)
            array_geo.radius_m = radius_cm / 100.0
            rep = evaluate_array_placement(
                geo,
                array_geo,
                min_clearance_m=min_clearance_mm / 1000.0,
                center_mode=center_mode,
                center_search_m=center_search_mm / 1000.0,
            )
            sx, sy = rep["array_center_shift_mm"]
            row = {
                "phantom_id": phantom_id,
                "pad_cells": int(pad),
                "radius_cm": float(radius_cm),
                "valid": bool(rep["valid"]),
                "n_oob": int(rep["n_oob"]),
                "n_in_tissue": int(rep["n_in_tissue"]),
                "min_clearance_mm": float(rep["min_clearance_m"] * 1000.0),
                "required_clearance_mm": float(min_clearance_mm),
                "suggested_radius_cm": float(rep["suggested_radius_m"] * 100.0),
                "center_shift_x_mm": float(sx),
                "center_shift_y_mm": float(sy),
                "center_mode": rep["center_mode_resolved"],
                "slice_desc": rep["slice_desc"],
            }
            all_rows.append(row)
            if row["valid"]:
                valid_rows.append(row)

    best = _pick_best(valid_rows)
    closest_failure = _pick_closest_failure(all_rows if not best else [])

    return {
        "phantom_id": phantom_id,
        "shape": list(phantom["shape"]),
        "voxel_mm": float(phantom["voxel_mm"]),
        "best": best,
        "closest_failure": closest_failure,
        "n_evaluated": len(all_rows),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit minimal radius/pad per phantom")
    parser.add_argument("--phantom-dir", default=os.path.join(_PROJECT_ROOT, "data", "uwcem"))
    parser.add_argument("--phantoms", nargs="*", default=None,
                        help="Optional explicit phantom IDs")
    parser.add_argument("--preset", default="umbmid",
                        choices=["umbmid", "maria", "mammowave"])
    parser.add_argument("--dx", type=float, default=0.001)
    parser.add_argument("--pads", default="40,50,60,70",
                        help="Comma-separated pad cell values")
    parser.add_argument("--radius-cm-min", type=float, default=7.0)
    parser.add_argument("--radius-cm-max", type=float, default=13.0)
    parser.add_argument("--radius-cm-step", type=float, default=0.5)
    parser.add_argument("--min-clearance-mm", type=float, default=3.0)
    parser.add_argument("--center-mode", default="ring_fit",
                        choices=["auto", "volume", "tissue_centroid",
                                 "skin_centroid", "ring_fit"])
    parser.add_argument("--center-search-mm", type=float, default=20.0)
    parser.add_argument("--output-json", default=None,
                        help="Optional JSON path for full report")
    args = parser.parse_args()

    pads = _parse_pads(args.pads)
    radii_cm = _radius_grid(args.radius_cm_min, args.radius_cm_max, args.radius_cm_step)
    if not radii_cm:
        raise ValueError("Empty radius grid. Check min/max/step values.")

    phantom_ids = args.phantoms or list_uwcem_phantoms(args.phantom_dir)
    if not phantom_ids:
        raise RuntimeError(f"No phantoms found in {args.phantom_dir}")

    print("=" * 84)
    print("WaveCare Array Geometry Audit")
    print(f"  phantom dir: {args.phantom_dir}")
    print(f"  preset: {args.preset}")
    print(f"  dx: {args.dx*1000:.2f} mm")
    print(f"  pads: {pads}")
    print(f"  radii: {radii_cm[0]:.2f} .. {radii_cm[-1]:.2f} cm (step {args.radius_cm_step:.2f})")
    print(f"  min clearance: {args.min_clearance_mm:.2f} mm")
    print(f"  center mode: {args.center_mode} (search {args.center_search_mm:.1f} mm)")
    print("=" * 84)

    results = []
    for pid in phantom_ids:
        res = audit_phantom(
            phantom_id=pid,
            phantom_dir=args.phantom_dir,
            preset_name=args.preset,
            dx=args.dx,
            pads=pads,
            radii_cm=radii_cm,
            min_clearance_mm=args.min_clearance_mm,
            center_mode=args.center_mode,
            center_search_mm=args.center_search_mm,
        )
        results.append(res)
        best = res["best"]
        if best is None:
            cf = res["closest_failure"]
            print(
                f"{pid}: NO VALID CONFIG in search grid "
                f"(closest: r={cf['radius_cm']:.2f}cm pad={cf['pad_cells']}, "
                f"oob={cf['n_oob']}, in_tissue={cf['n_in_tissue']}, "
                f"clear={cf['min_clearance_mm']:.2f}mm)"
            )
        else:
            print(
                f"{pid}: r={best['radius_cm']:.2f}cm pad={best['pad_cells']} "
                f"shift=({best['center_shift_x_mm']:+.1f},{best['center_shift_y_mm']:+.1f})mm "
                f"clear={best['min_clearance_mm']:.2f}mm"
            )

    valid_rows = [r["best"] for r in results if r["best"] is not None]
    print("-" * 84)
    print(f"Valid phantoms: {len(valid_rows)}/{len(results)}")
    if valid_rows:
        worst_radius = max(v["radius_cm"] for v in valid_rows)
        worst_pad = max(v["pad_cells"] for v in valid_rows)
        print(f"Worst-case among valid best configs: radius={worst_radius:.2f}cm, pad={worst_pad}")

    report = {
        "config": {
            "phantom_dir": args.phantom_dir,
            "preset": args.preset,
            "dx": args.dx,
            "pads": pads,
            "radius_cm_min": args.radius_cm_min,
            "radius_cm_max": args.radius_cm_max,
            "radius_cm_step": args.radius_cm_step,
            "min_clearance_mm": args.min_clearance_mm,
            "center_mode": args.center_mode,
            "center_search_mm": args.center_search_mm,
        },
        "results": results,
    }

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report: {args.output_json}")


if __name__ == "__main__":
    main()

