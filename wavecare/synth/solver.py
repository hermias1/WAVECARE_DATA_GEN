"""gprMax FDTD solver wrapper.

Generates input files from scene + acquisition geometry,
runs gprMax, and collects results.
"""

import os
import time
import numpy as np
import h5py
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d


def prepare_2d_geometry(scene, dx=0.001, pad_cells=40, slice_idx=None):
    """Extract 2D axial slice from a 3D scene and prepare for gprMax.

    Parameters
    ----------
    scene : dict
        Output from scenes.generate_scene().
    dx : float
        Target cell size in meters.
    pad_cells : int
        Padding cells for coupling medium on each side.
    slice_idx : int or None
        Axial slice index. None = tumor center or midpoint.

    Returns
    -------
    dict with keys: labels_2d, domain_m, center_m, geo_shape
    """
    mtype = scene["mtype"]
    voxel_mm = scene["voxel_mm"]

    # Choose slice
    if slice_idx is None:
        if scene.get("tumor_info"):
            # Use tumor center
            tc = scene["tumor_info"]["center_mm"]
            slice_idx = int(tc[2] / voxel_mm)
            slice_idx = min(max(slice_idx, 0), mtype.shape[2] - 1)
        else:
            slice_idx = mtype.shape[2] // 2

    mtype_2d = mtype[:, :, slice_idx].copy()

    # Resample to target resolution
    src_mm = voxel_mm
    tgt_mm = dx * 1000
    if abs(src_mm - tgt_mm) > 0.01:
        scale = src_mm / tgt_mm
        new_nx = int(mtype_2d.shape[0] * scale)
        new_ny = int(mtype_2d.shape[1] * scale)
        x_idx = (np.arange(new_nx) / scale).astype(int).clip(0, mtype_2d.shape[0]-1)
        y_idx = (np.arange(new_ny) / scale).astype(int).clip(0, mtype_2d.shape[1]-1)
        mtype_2d = mtype_2d[np.ix_(x_idx, y_idx)]

    nx, ny = mtype_2d.shape

    # Pad for coupling medium
    padded = np.zeros((nx + 2*pad_cells, ny + 2*pad_cells), dtype=np.int8)
    padded[pad_cells:pad_cells+nx, pad_cells:pad_cells+ny] = mtype_2d

    domain_x = padded.shape[0] * dx
    domain_y = padded.shape[1] * dx
    center_x = (pad_cells + nx / 2) * dx
    center_y = (pad_cells + ny / 2) * dx

    return {
        "labels_2d": padded,
        "domain_m": (domain_x, domain_y),
        "center_m": (center_x, center_y),
        "geo_shape": padded.shape,
        "dx": dx,
        "pad_cells": pad_cells,
        "slice_idx": slice_idx,
    }


def write_geometry_files(geo_info, work_dir):
    """Write HDF5 geometry and materials file for gprMax.

    Returns (geo_path, mat_path).
    """
    os.makedirs(work_dir, exist_ok=True)
    dx = geo_info["dx"]

    # HDF5 geometry
    geo_path = os.path.join(work_dir, "breast_geo.h5")
    data_3d = geo_info["labels_2d"].astype(np.int16)[:, :, np.newaxis]
    with h5py.File(geo_path, 'w') as f:
        f.attrs['dx_dy_dz'] = (dx, dx, dx)
        f.create_dataset('data', data=data_3d, dtype='int16')

    # Materials file (Debye 1-pole, 11 tissue types)
    mat_path = os.path.join(work_dir, "breast_mat.txt")
    mat_lines = [
        # idx 0: coupling medium
        "#material: 10.0 0.01 1 0 coupling_medium",
        # idx 1: skin
        "#material: 4.0 0.0002 1 0 skin",
        "#add_dispersion_debye: 1 32.0 7.234e-12 skin",
        # idx 2: muscle
        "#material: 4.0 0.20 1 0 muscle",
        "#add_dispersion_debye: 1 50.0 7.234e-12 muscle",
        # idx 3-5: fat (low/med/high water)
        "#material: 2.55 0.020 1 0 fat_1",
        "#add_dispersion_debye: 1 1.20 13.0e-12 fat_1",
        "#material: 2.55 0.036 1 0 fat_2",
        "#add_dispersion_debye: 1 1.71 13.0e-12 fat_2",
        "#material: 3.0 0.083 1 0 fat_3",
        "#add_dispersion_debye: 1 3.65 13.0e-12 fat_3",
        # idx 6: transitional
        "#material: 5.5 0.304 1 0 transitional",
        "#add_dispersion_debye: 1 16.55 13.0e-12 transitional",
        # idx 7-9: fibroglandular
        "#material: 9.9 0.462 1 0 fibro_1",
        "#add_dispersion_debye: 1 26.60 13.0e-12 fibro_1",
        "#material: 13.81 0.738 1 0 fibro_2",
        "#add_dispersion_debye: 1 35.55 13.0e-12 fibro_2",
        "#material: 6.15 0.809 1 0 fibro_3",
        "#add_dispersion_debye: 1 48.26 13.0e-12 fibro_3",
        # idx 10: tumor
        "#material: 14.0 0.90 1 0 tumor",
        "#add_dispersion_debye: 1 42.0 13.0e-12 tumor",
    ]
    with open(mat_path, 'w') as f:
        f.write('\n'.join(mat_lines) + '\n')

    return geo_path, mat_path


def generate_gprmax_inputs(geo_info, array_geo, work_dir, time_window=8e-9):
    """Generate gprMax .in files for all Tx-Rx pairs.

    Parameters
    ----------
    geo_info : dict
        From prepare_2d_geometry().
    array_geo : ArrayGeometry
        Acquisition geometry.
    work_dir : str
        Directory for .in files.
    time_window : float
        Simulation time window in seconds.

    Returns
    -------
    list of (input_path, tx_idx, rx_idx)
    """
    dx = geo_info["dx"]
    domain_x, domain_y = geo_info["domain_m"]
    offset = geo_info["pad_cells"] * dx

    # Shift array center to breast center in domain
    cx, cy = geo_info["center_m"]

    positions = array_geo.antenna_positions()
    # Shift positions to domain coordinates
    positions[:, 0] += cx
    positions[:, 1] += cy

    pairs = array_geo.tx_rx_pairs()
    inputs = []

    for pair_idx, (tx_idx, rx_idx) in enumerate(pairs):
        tx_x = round(positions[tx_idx, 0] / dx) * dx
        tx_y = round(positions[tx_idx, 1] / dx) * dx
        rx_x = round(positions[rx_idx, 0] / dx) * dx
        rx_y = round(positions[rx_idx, 1] / dx) * dx

        lines = [
            f"#title: scan pair {pair_idx} (tx={tx_idx} rx={rx_idx})",
            f"#domain: {domain_x:.6f} {domain_y:.6f} {dx:.6f}",
            f"#dx_dy_dz: {dx:.6f} {dx:.6f} {dx:.6f}",
            f"#time_window: {time_window:.2e}",
            "",
            "#material: 10.0 0.01 1 0 bg_coupling",
            f"#box: 0 0 0 {domain_x:.6f} {domain_y:.6f} {dx:.6f} bg_coupling",
            "",
            f"#geometry_objects_read: {offset:.6f} {offset:.6f} 0 "
            "breast_geo.h5 breast_mat.txt",
            "",
            "#waveform: ricker 1 4.5e9 uwb_pulse",
            f"#hertzian_dipole: z {tx_x:.6f} {tx_y:.6f} 0 uwb_pulse",
            "",
            f"#rx: {rx_x:.6f} {rx_y:.6f} 0",
        ]

        path = os.path.join(work_dir, f"pair_{pair_idx:04d}.in")
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        inputs.append((path, tx_idx, rx_idx))

    return inputs


def run_simulation(input_file):
    """Run a single gprMax simulation."""
    from gprMax.gprMax import api
    api(input_file)


def run_scan(inputs, verbose=True, max_failures=None):
    """Run all simulations sequentially.

    Parameters
    ----------
    inputs : list of (path, tx_idx, rx_idx)
    verbose : bool
    max_failures : int or None
        Maximum allowed failures before aborting. None = 10% of total.

    Returns
    -------
    float : total time in seconds

    Raises
    ------
    RuntimeError
        If too many simulations fail.
    """
    n = len(inputs)
    if max_failures is None:
        max_failures = max(1, n // 10)

    t0 = time.time()
    n_failures = 0

    for i, (infile, tx, rx) in enumerate(inputs):
        ts = time.time()
        try:
            run_simulation(infile)
        except Exception as e:
            n_failures += 1
            if verbose:
                print(f"  FAILED pair {i} (tx={tx}, rx={rx}): {e}")
            if n_failures > max_failures:
                raise RuntimeError(
                    f"Aborting: {n_failures}/{i+1} simulations failed "
                    f"(threshold: {max_failures})"
                ) from e
            continue

        if verbose:
            elapsed = time.time() - ts
            if i == 0:
                print(f"  Pair 0: {elapsed:.1f}s "
                      f"(est. total: {elapsed * n / 60:.1f} min)")
            elif (i + 1) % max(1, n // 6) == 0:
                total = time.time() - t0
                eta = total / (i + 1) * (n - i - 1) / 60
                print(f"  [{i+1}/{n}] {elapsed:.1f}s/sim, ETA: {eta:.1f} min")

    if verbose and n_failures > 0:
        print(f"  Warning: {n_failures}/{n} simulations failed")

    return time.time() - t0


def collect_results(work_dir, inputs, target_freqs_hz=None):
    """Collect simulation outputs into arrays.

    Parameters
    ----------
    work_dir : str
    inputs : list of (path, tx_idx, rx_idx)
    target_freqs_hz : ndarray or None
        If given, interpolate FFT to these frequencies (for UMBMID compat).

    Returns
    -------
    dict with 'td_data', 'fd_data', 'dt', 'freqs_hz', 'pairs'
    """
    all_ez = []
    dt = None

    for path, _, _ in inputs:
        out = path.replace('.in', '.out')
        if not os.path.exists(out):
            all_ez.append(None)
            continue
        with h5py.File(out, 'r') as f:
            ez = f['rxs']['rx1']['Ez'][:]
            if dt is None:
                dt = f.attrs['dt']
            all_ez.append(ez)

    # Pad to uniform length
    valid = [e for e in all_ez if e is not None]
    if not valid:
        raise RuntimeError("No valid simulation outputs found")

    n_missing = len(all_ez) - len(valid)
    if n_missing > 0:
        import warnings
        warnings.warn(
            f"{n_missing}/{len(all_ez)} simulation outputs missing. "
            f"Corresponding rows will be zero-filled."
        )

    max_len = max(len(e) for e in valid)
    n_pairs = len(inputs)
    td_data = np.zeros((n_pairs, max_len))
    for i, ez in enumerate(all_ez):
        if ez is not None:
            td_data[i, :len(ez)] = ez

    # FFT
    freqs = fftfreq(max_len, dt)
    fd_full = fft(td_data, axis=1)

    # Interpolate to target frequencies if requested
    if target_freqs_hz is not None:
        pos = freqs > 0
        freqs_pos = freqs[pos]
        fd_pos = fd_full[:, pos]

        fd_data = np.zeros((n_pairs, len(target_freqs_hz)), dtype=complex)
        for i in range(n_pairs):
            f_real = interp1d(freqs_pos, fd_pos[i].real,
                              fill_value=0, bounds_error=False)
            f_imag = interp1d(freqs_pos, fd_pos[i].imag,
                              fill_value=0, bounds_error=False)
            fd_data[i] = f_real(target_freqs_hz) + 1j * f_imag(target_freqs_hz)

        return {
            "td_data": td_data,
            "fd_data": fd_data,
            "dt": dt,
            "freqs_hz": target_freqs_hz,
            "pairs": [(tx, rx) for _, tx, rx in inputs],
        }

    return {
        "td_data": td_data,
        "fd_data": fd_full,
        "dt": dt,
        "freqs_hz": freqs,
        "pairs": [(tx, rx) for _, tx, rx in inputs],
    }
