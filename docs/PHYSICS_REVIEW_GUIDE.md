# Physics Review Guide

This document is intended for collaborators with a physics/EM background who
want to review whether the WaveCare synthetic pipeline is physically reasonable.

## 1. What This Repository Simulates

WaveCare generates synthetic microwave breast-imaging scans by combining:
- voxelized anatomical breast phantoms (UWCEM),
- dispersive tissue dielectric models (Debye/Cole-Cole family),
- full-wave FDTD propagation in gprMax.

Main objective:
- produce labeled synthetic data for ML (`tumor` vs `no tumor`) while staying
  close to UM-BMID acquisition conventions.

## 2. What Is "UM-BMID-like" Here

- Frequency band: `1-8 GHz`
- Frequency samples: `1001`
- Angular positions: `72` (5-degree step)
- Bistatic offset: `+60 degree`
- Start angle: `-130 degree`

Code references:
- `wavecare/acqui/presets.py`
- `wavecare/acqui/geometry.py`

## 3. Core Pipeline (Conceptual)

1. Load voxel phantom (`mtype.zip`, `breastInfo.txt`).
2. Insert tumor (for positive samples).
3. Resample phantom to FDTD cell size (`dx`).
4. Pad with coupling medium.
5. Generate one gprMax simulation per Tx/Rx pair.
6. Collect time-domain receiver traces (`Ez(t)`).
7. FFT to frequency domain and apply reference subtraction.

Code references:
- `wavecare/synth/phantoms.py`
- `wavecare/synth/scenes.py`
- `wavecare/synth/solver.py`
- `wavecare/synth/export.py`

## 4. Very Important Distinction for Review

The solver currently models:
- source: Hertzian dipole (`#hertzian_dipole`)
- receiver: point electric field probe (`#rx`, collecting `Ez`)

So the measured quantity is:
- calibrated field-like quantity derived from `Ez`, not direct VNA port
  `S21` from full antenna models.

This can be acceptable for pattern-learning tasks, but not equivalent to
full port-calibrated measurement physics.

## 5. Current Safety Guardrails

Antenna placement validation is now enforced before `.in` generation:
- fails if antenna points are out of domain,
- fails if antenna points are inside tissue,
- optional minimum clearance to tissue.

Code references:
- `wavecare/synth/solver.py` (`_check_antenna_positions`)
- options in scripts:
  - `--radius-cm`
  - `--min-clearance-mm`

## 6. Suggested Reading Order

1. `docs/PHYSICS_MODEL_AND_ASSUMPTIONS.md`
2. `docs/PHYSICS_VALIDATION_CHECKLIST.md`
3. `wavecare/synth/solver.py`
4. `wavecare/synth/dielectrics.py`

## 7. Minimal Commands for a Reviewer

Run tests:

```bash
python3 -m pytest -q tests
```

Try a 2D synthetic scan with explicit geometry constraints:

```bash
python3 scripts/generate_scan.py \
  --phantom 071904 \
  --preset umbmid \
  --radius-cm 11 \
  --min-clearance-mm 3 \
  --tumor-mm 12
```

For 3D cluster pipeline prep (no simulation launch in this command):

```bash
python3 scripts/slurm_scan.py prepare \
  --phantom 071904 \
  --preset umbmid \
  --radius-cm 11 \
  --min-clearance-mm 3 \
  --work-dir /tmp/wavecare_review
```

If you get an "outside simulation domain" placement error, increase padding
(`--pad`) and/or reduce `--radius-cm`.

Numerical sensitivity (dx vs fmax):

```bash
python3 scripts/physics_sensitivity.py --dx-mm 1 --fmin-ghz 1 --fmax-ghz 8
```

Quick synthetic-vs-real metric report (if UM-BMID files are available):

```bash
python3 scripts/compare_with_umbmid.py \
  --synthetic /path/to/synthetic_scan.npz \
  --real /path/to/fd_data_s11_emp.pickle
```

Batch geometry audit (minimal valid radius/pad by phantom):

```bash
python3 scripts/audit_array_geometry.py \
  --phantom-dir /path/to/uwcem \
  --preset umbmid \
  --dx 0.001 \
  --pads 40,50,60,70 \
  --radius-cm-min 7 \
  --radius-cm-max 13 \
  --radius-cm-step 0.5 \
  --min-clearance-mm 3 \
  --center-mode ring_fit
```
