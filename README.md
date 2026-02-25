# WaveCare Data Generation

Synthetic microwave imaging (MWI) data generation pipeline for breast tumor detection research.

Generates realistic simulated MWI scans by combining anatomically accurate breast phantoms (MRI-derived, heterogeneous tissues) with FDTD electromagnetic simulation. Designed to bridge the gap between phantom-based lab data and clinical in-vivo measurements.

## Key Features

- **Anatomically realistic phantoms**: 10 tissue types from MRI-derived UWCEM models (0.5 mm resolution)
- **Dispersive dielectric properties**: Debye 1-pole models based on Lazebnik et al. (2007)
- **Parametric acquisition geometry**: configurable antenna arrays (UMBMID, MARIA M5, Mammowave, custom)
- **FDTD simulation**: full-wave electromagnetic solving via gprMax
- **Domain randomization**: variable tumor size/position, tissue perturbation, noise injection
- **Flexible output**: time-domain and frequency-domain, compatible with UMBMID format

## Architecture

```
wavecare/
  synth/             Synthetic data generation core
    dielectrics.py     Cole-Cole & Debye tissue models (Lazebnik, Gabriel)
    phantoms.py        UWCEM & Pelicano phantom loaders
    tumors.py          Tumor insertion (sphere, ellipsoid)
    scenes.py          Scene assembly + domain randomization
    noise.py           AWGN, phase noise, amplitude noise
    solver.py          gprMax FDTD wrapper
    export.py          Dataset export (npz)
  acqui/             Acquisition configuration
    geometry.py        Parametric circular antenna array
    presets.py         UMBMID Gen2, MARIA M5, Mammowave
  eval/              Evaluation & benchmarking
    compare.py         Synthetic vs real data metrics
```

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Install gprMax (FDTD solver)

Requires Homebrew GCC on macOS:

```bash
brew install gcc
bash scripts/install_gprmax.sh
```

### 3. Download phantom data

Download UWCEM phantoms from https://uwcem.ece.wisc.edu/phantomRepository.html and place in `data/uwcem/`:

```
data/uwcem/
  071904/          ACR Class 1 (fatty)
    breastInfo.txt
    mtype.zip
  062204/          ACR Class 3 (heterogeneously dense)
    breastInfo.txt
    mtype.zip
```

### 4. Generate a scan

```bash
# UMBMID-compatible scan (72 angles, bistatic)
python scripts/generate_scan.py --phantom 071904 --preset umbmid --tumor-mm 12

# MARIA-like clinical scan (60 antennas, multistatic)
python scripts/generate_scan.py --phantom 071904 --preset maria --tumor-mm 8

# Reference scan without tumor
python scripts/generate_scan.py --phantom 071904 --preset umbmid --no-tumor

# Physics-safe geometry defaults for review
python scripts/generate_scan.py --phantom 071904 --preset umbmid \
  --radius-cm 11 --min-clearance-mm 3 --center-mode ring_fit
```

Output is saved as `output/scan_<phantom>_<preset>/scan_data.npz` containing:
- `td_data`: time-domain signals, shape `(n_measurements, n_timesteps)`
- `fd_data`: frequency-domain signals, shape `(n_measurements, n_freqs)`, complex
- `freqs_hz`: frequency axis
- `dt`: time step
- Metadata: phantom ID, tumor parameters, acquisition config

## Physics Review Pack

For external collaborators (especially physicists), start with:

- `docs/PHYSICS_REVIEW_GUIDE.md`
- `docs/PHYSICS_MODEL_AND_ASSUMPTIONS.md`
- `docs/PHYSICS_VALIDATION_CHECKLIST.md`
- `docs/RELECTURE_PHYSIQUE_FR.md` (French quick entry point)

Important interpretation note:
- Current simulations use a Hertzian dipole source and point-field receiver (`Ez`),
  then frequency-domain conversion and reference subtraction.
- This is not a full port-impedance antenna model producing direct VNA `S21`.

Additional analysis scripts:
- `scripts/physics_sensitivity.py` (dx/frequency numerical sensitivity)
- `scripts/compare_with_umbmid.py` (quick synthetic-vs-real metric report)
- `scripts/audit_array_geometry.py` (find minimal valid radius/pad per phantom)

## Acquisition Presets

| Preset | Antennas | Mode | Freq (GHz) | Measurements |
|--------|----------|------|------------|--------------|
| `umbmid` | 72 | Bistatic (+60 deg) | 1-8 | 72 |
| `maria` | 60 | Multistatic | 3-8 | 1770 |
| `mammowave` | 72 | Bistatic (180 deg) | 1-9 | 72 |

## Tissue Dielectric Properties

Based on Lazebnik et al. 2007 (breast) and Gabriel et al. 1996 (skin, muscle).

| Tissue | eps_r @ 6 GHz | sigma (S/m) | Source |
|--------|--------------|-------------|--------|
| Fat (low water) | ~4 | 0.02 | Lazebnik Group 3 |
| Transitional | ~22 | 0.30 | Lazebnik Group 2 |
| Fibroglandular | ~45 | 0.74 | Lazebnik Group 1 |
| Tumor (malignant) | ~55 | 0.90 | Lazebnik 2007b |
| Skin | ~36 | 0.0002 | Gabriel 1996 |

## Performance

On Apple M5 (10 cores), 2D TMz simulation at 1 mm resolution:
- ~7 s per simulation (235 x 257 cells, 3393 time steps)
- ~8 min for a full 72-angle UMBMID scan
- ~2 hours for a 1770-pair MARIA multistatic scan (estimated)

## References

- Lazebnik et al., *Phys. Med. Biol.* 52:2637 (2007) - Normal breast tissue dielectrics
- Lazebnik et al., *Phys. Med. Biol.* 52:6093 (2007) - Malignant breast tissue dielectrics
- Gabriel et al., *Phys. Med. Biol.* 41:2271 (1996) - Tissue dielectric database
- UWCEM Numerical Breast Phantom Repository - https://uwcem.ece.wisc.edu/
- Warren et al., *Computer Physics Communications* 209:163-170 (2016) - gprMax
- Baran et al., *IEEE Access* 10 (2022) - UM-BMID dataset

## Context

Part of the WaveCare project (SogetiLabs / Capgemini). See `paper/` for the ICPR 2026 submission on modular deep learning for breast tumor detection from microwave imaging.

## License

MIT
