# Physics Validation Checklist

Use this checklist to review physical fidelity before large data generation.

## A. Acquisition Geometry

- [ ] Angular sampling matches intended system (`72` positions, `5 degree` step).
- [ ] Start angle and bistatic offset match target setup (`-130 degree`, `+60 degree` for UM-BMID-like).
- [ ] Antennas are outside breast tissue in simulation domain.
- [ ] Minimum antenna-to-tissue clearance is explicitly enforced.

How to enforce in this repo:
- use `--radius-cm` and `--min-clearance-mm` in scan scripts.
- pre-run will fail if placement is invalid.

## B. Frequency and Time Settings

- [ ] Frequency range and sampling align with target (`1-8 GHz`, `1001` points when UM-BMID-like).
- [ ] Time window is sufficient for direct and relevant delayed paths.
- [ ] IF any down-banding is used (e.g., `1-5 GHz`), it is documented.

## C. Material Modeling

- [ ] Tissue dielectric values reviewed against source literature.
- [ ] Debye/Cole-Cole approximation choice is justified for intended bandwidth.
- [ ] Any perturbation/randomization level is physically motivated and documented.

## D. Numerical Resolution

- [ ] `dx` yields acceptable cells-per-wavelength in highest-permittivity tissues.
- [ ] Numerical dispersion risk at `f_max` is assessed.
- [ ] Chosen tradeoff (`dx` vs runtime) is documented.

## E. Measured Quantity Interpretation

- [ ] Team agrees that simulated quantity is `Ez`-derived, not true port-level `S21`.
- [ ] Calibration method (`tumor - reference`) is clearly stated.
- [ ] Claims in papers/presentations match actual simulated observable.

## F. Internal Consistency Checks

- [ ] Tumor and reference scans use consistent geometry assumptions.
- [ ] 2D vs 3D usage is explicit (no mixing claims).
- [ ] Missing outputs / failed sims are tracked and reported.

## G. Minimum Documentation Before Sharing Results

- [ ] Include run command and parameters (`dx`, `pad`, `radius`, `clearance`, `preset`).
- [ ] Include software commit hash.
- [ ] Include phantom IDs used and class distribution.
- [ ] Include known limitations section.

## Recommended Baseline for Current Repo (Operational)

- `preset`: `umbmid`
- `dx`: `1 mm` (if runtime-limited)
- `center mode`: `ring_fit` (or `auto`, which resolves to ring_fit)
- `radius`: choose from phantom audit (often ~10-12 cm depending on phantom/pad)
- `min clearance`: `3 mm`
- if out-of-bounds occurs, increase `pad` and/or reduce radius
- then re-evaluate with phantom-by-phantom diagnostics before production runs
