# Physics Model and Assumptions

## 1. Electromagnetic Solver

- Solver backend: gprMax (FDTD on Yee grid).
- Time-domain simulation, then FFT post-processing.
- 2D mode:
  - thin `z` domain (`#domain: x y dx` with one-cell thickness),
  - useful for rapid experiments, not full 3D propagation realism.
- 3D mode:
  - full voxel volume with 3D ring placement at one `z` level.

Code references:
- `wavecare/synth/solver.py`

## 2. Geometry and Phantoms

- Source anatomical models: UWCEM MRI-derived numerical phantoms.
- Native resolution: typically `0.5 mm` voxels.
- Tissue labels mapped to integer classes:
  - background, skin, muscle, adipose subtypes, transitional,
    fibroglandular subtypes.
- Optional tumor insertion:
  - sphere/ellipsoid, random valid interior position.

Code references:
- `wavecare/synth/phantoms.py`
- `wavecare/synth/tumors.py`
- `wavecare/synth/scenes.py`

## 3. Dielectric Parameterization

- Primary implementation in solver uses Debye-like one-pole material cards
  for gprMax (`#material` + `#add_dispersion_debye`).
- Parameters are aligned with Lazebnik/Gabriel-inspired values used in code.
- Optional Gaussian perturbation can be applied to dielectric parameters for
  synthetic variability (especially negative/reference modes).

Code references:
- `wavecare/synth/solver.py` (`write_geometry_files`)
- `wavecare/synth/dielectrics.py`

## 4. Acquisition Modeling

- Parametric circular array.
- UM-BMID-like preset:
  - `72` positions,
  - start angle `-130 degree`,
  - bistatic offset `+60 degree`,
  - `1-8 GHz`, `1001` points.

Code references:
- `wavecare/acqui/presets.py`
- `wavecare/acqui/geometry.py`

## 5. Excitation and Observation

Per pair simulation currently uses:
- waveform: `ricker`, center frequency `4.5 GHz`,
- transmitter: `#hertzian_dipole`,
- receiver: point probe `#rx`,
- collected signal: `Ez(t)`.

This is not a full port-level antenna model. It is a simplified EM source/receiver
representation that captures propagation/scattering patterns but not full antenna
impedance/network behavior.

Code references:
- `wavecare/synth/solver.py` (`generate_gprmax_inputs*`, `collect_results`)

## 6. Calibration and Export

- Frequency-domain conversion by FFT + interpolation to target frequencies.
- Calibration step: subtraction (`tumor - reference`).
- Export format compatible with UM-BMID style tensor organization.

Code references:
- `wavecare/synth/export.py`
- `wavecare/synth/solver.py` (`collect_results`)

## 7. Practical Numerical Assumptions

- Spatial discretization is user-defined (`dx`), default often `1 mm`.
- Padding in coupling medium is configurable (`pad_cells`).
- Time windows:
  - 2D default around `8 ns`,
  - 3D default around `12 ns`.

Numerical accuracy depends strongly on:
- cells per wavelength in high-permittivity tissues,
- dispersion error at upper frequencies.

## 8. Current High-Impact Approximations

1. `Ez` proxy instead of full `S21` port-calibrated response.
2. Simplified source/receiver antennas (Hertzian dipole + point probe).
3. One-pole Debye simplification for broad tissue dispersion.
4. 2D mode as a computational proxy for 3D physics.

These are acceptable for some synthetic ML scenarios, but should be explicitly
acknowledged in any physics-focused publication.

