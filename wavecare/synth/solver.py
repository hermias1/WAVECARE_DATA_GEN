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
from scipy.ndimage import distance_transform_edt


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


def prepare_3d_geometry(scene, dx=0.001, pad_cells=50):
    """Prepare full 3D geometry from a scene for gprMax.

    Parameters
    ----------
    scene : dict
        Output from scenes.generate_scene().
    dx : float
        Target cell size in meters.
    pad_cells : int
        Padding cells for coupling medium on each side.

    Returns
    -------
    dict with keys: labels_3d, domain_m, center_m, geo_shape, dx, pad_cells
    """
    mtype = scene["mtype"]
    voxel_mm = scene["voxel_mm"]

    # Resample to target resolution
    src_mm = voxel_mm
    tgt_mm = dx * 1000
    if abs(src_mm - tgt_mm) > 0.01:
        scale = src_mm / tgt_mm
        new_nx = int(mtype.shape[0] * scale)
        new_ny = int(mtype.shape[1] * scale)
        new_nz = int(mtype.shape[2] * scale)
        x_idx = (np.arange(new_nx) / scale).astype(int).clip(0, mtype.shape[0]-1)
        y_idx = (np.arange(new_ny) / scale).astype(int).clip(0, mtype.shape[1]-1)
        z_idx = (np.arange(new_nz) / scale).astype(int).clip(0, mtype.shape[2]-1)
        mtype = mtype[np.ix_(x_idx, y_idx, z_idx)]

    nx, ny, nz = mtype.shape

    # Pad for coupling medium
    padded = np.zeros(
        (nx + 2*pad_cells, ny + 2*pad_cells, nz + 2*pad_cells), dtype=np.int8)
    padded[pad_cells:pad_cells+nx,
           pad_cells:pad_cells+ny,
           pad_cells:pad_cells+nz] = mtype

    domain_x = padded.shape[0] * dx
    domain_y = padded.shape[1] * dx
    domain_z = padded.shape[2] * dx
    center_x = (pad_cells + nx / 2) * dx
    center_y = (pad_cells + ny / 2) * dx
    center_z = (pad_cells + nz / 2) * dx

    return {
        "labels_3d": padded,
        "domain_m": (domain_x, domain_y, domain_z),
        "center_m": (center_x, center_y, center_z),
        "geo_shape": padded.shape,
        "dx": dx,
        "pad_cells": pad_cells,
    }


def write_geometry_files(geo_info, work_dir, perturbation=0.0, rng=None):
    """Write HDF5 geometry and materials file for gprMax.

    Handles both 2D (labels_2d) and 3D (labels_3d) geometry.

    Parameters
    ----------
    geo_info : dict
    work_dir : str
    perturbation : float
        If > 0, jitter Debye parameters by this fraction (e.g. 0.05 = 5%).
        Useful for generating distinct reference scans for negative samples.
    rng : numpy.random.Generator or None
        Required if perturbation > 0.

    Returns (geo_path, mat_path).
    """
    os.makedirs(work_dir, exist_ok=True)
    dx = geo_info["dx"]

    # HDF5 geometry — detect 2D vs 3D
    geo_path = os.path.join(work_dir, "breast_geo.h5")
    if "labels_3d" in geo_info:
        data = geo_info["labels_3d"].astype(np.int16)
    else:
        data = geo_info["labels_2d"].astype(np.int16)[:, :, np.newaxis]
    with h5py.File(geo_path, 'w') as f:
        f.attrs['dx_dy_dz'] = (dx, dx, dx)
        f.create_dataset('data', data=data, dtype='int16')

    # Base Debye parameters: (eps_inf, sigma, delta_eps, tau, name)
    _TISSUES = [
        (10.0,  0.01,   None,     None,      "coupling_medium"),
        (4.0,   0.0002, 32.0,     7.234e-12, "skin"),
        (4.0,   0.20,   50.0,     7.234e-12, "muscle"),
        (2.55,  0.020,  1.20,     13.0e-12,  "fat_1"),
        (2.55,  0.036,  1.71,     13.0e-12,  "fat_2"),
        (3.0,   0.083,  3.65,     13.0e-12,  "fat_3"),
        (5.5,   0.304,  16.55,    13.0e-12,  "transitional"),
        (9.9,   0.462,  26.60,    13.0e-12,  "fibro_1"),
        (13.81, 0.738,  35.55,    13.0e-12,  "fibro_2"),
        (6.15,  0.809,  48.26,    13.0e-12,  "fibro_3"),
        (14.0,  0.90,   42.0,     13.0e-12,  "tumor"),
    ]

    def _jitter(val, rng, pct):
        """Apply Gaussian jitter: val * (1 + N(0, pct))."""
        return val * (1 + rng.normal(0, pct))

    mat_lines = []
    for eps_inf, sigma, delta_eps, tau, name in _TISSUES:
        if perturbation > 0 and rng is not None:
            eps_inf = _jitter(eps_inf, rng, perturbation)
            sigma = abs(_jitter(sigma, rng, perturbation))
            if delta_eps is not None:
                delta_eps = _jitter(delta_eps, rng, perturbation)

        mat_lines.append(f"#material: {eps_inf:.4f} {sigma:.6f} 1 0 {name}")
        if delta_eps is not None:
            mat_lines.append(
                f"#add_dispersion_debye: 1 {delta_eps:.4f} {tau:.3e} {name}")

    mat_path = os.path.join(work_dir, "breast_mat.txt")
    with open(mat_path, 'w') as f:
        f.write('\n'.join(mat_lines) + '\n')

    return geo_path, mat_path


def _get_placement_slice(geo_info, ant_z_m=None):
    """Return the 2D label slice used for antenna placement checks."""
    dx = geo_info["dx"]
    if "labels_3d" in geo_info:
        labels = geo_info["labels_3d"]
        if ant_z_m is None:
            raise ValueError("ant_z_m must be provided for 3D antenna validation")
        iz = int(round(ant_z_m / dx))
        if iz < 0 or iz >= labels.shape[2]:
            raise ValueError(
                f"Antenna z index out of bounds: {iz} not in [0, {labels.shape[2] - 1}]"
            )
        return labels[:, :, iz], f"3D z={ant_z_m * 1000:.1f}mm"
    return geo_info["labels_2d"], f"2D slice_idx={geo_info.get('slice_idx', 'n/a')}"


def _antenna_positions_with_center(array_geo, cx, cy):
    """Compute antenna positions with center already in domain coordinates."""
    positions = array_geo.antenna_positions()
    positions[:, 0] += cx
    positions[:, 1] += cy
    return positions


def _placement_score(slice_2d, dx, positions_xy, clearance_map, min_clearance_m):
    """Score ring placement: lower is better (0 means perfect)."""
    ix = np.rint(positions_xy[:, 0] / dx).astype(int)
    iy = np.rint(positions_xy[:, 1] / dx).astype(int)

    in_bounds = (
        (ix >= 0) & (ix < slice_2d.shape[0]) &
        (iy >= 0) & (iy < slice_2d.shape[1])
    )
    n_oob = int(np.sum(~in_bounds))
    if n_oob > 0:
        # Very high penalty for out-of-bounds placements.
        return 1e12 + n_oob * 1e9, n_oob, 0, 0.0

    sampled_labels = slice_2d[ix, iy]
    n_in_tissue = int(np.sum(sampled_labels > 0))
    clearances = clearance_map[ix, iy]
    min_clearance_found = float(np.min(clearances)) if clearances.size else float("inf")
    clearance_deficit = np.maximum(0.0, min_clearance_m - clearances)

    # Hierarchical objective:
    # 1) avoid tissue intersection
    # 2) satisfy clearance
    # 3) maximize margin via residual term
    score = (
        n_in_tissue * 1e9
        + float(np.sum(clearance_deficit)) / max(dx, 1e-12) * 1e4
        + max(0.0, min_clearance_m - min_clearance_found) * 1e3
    )
    return score, n_oob, n_in_tissue, min_clearance_found


def _resolve_array_center(
    geo_info,
    array_geo,
    center_mode="auto",
    min_clearance_m=0.0,
    ant_z_m=None,
    center_search_m=0.02,
):
    """Resolve array center in domain coordinates.

    Modes:
    - volume: use geometric center from padded grid.
    - tissue_centroid: centroid of tissue mask (label > 0).
    - skin_centroid: centroid of skin mask (label == 1), falls back to tissue.
    - ring_fit: local grid-search minimizing tissue hits / clearance deficits.
    - auto: ring_fit with fallback to volume.
    """
    dx = geo_info["dx"]
    base_cx, base_cy = geo_info["center_m"][:2]
    slice_2d, slice_desc = _get_placement_slice(geo_info, ant_z_m=ant_z_m)
    tissue_mask = slice_2d > 0
    clearance_map = distance_transform_edt(~tissue_mask) * dx

    def _centroid(mask):
        idx = np.argwhere(mask)
        if idx.size == 0:
            return None
        return float(np.mean(idx[:, 0]) * dx), float(np.mean(idx[:, 1]) * dx)

    resolved_mode = center_mode
    if center_mode == "auto":
        resolved_mode = "ring_fit"

    if resolved_mode == "volume":
        cx, cy = base_cx, base_cy
    elif resolved_mode == "tissue_centroid":
        c = _centroid(tissue_mask)
        cx, cy = c if c is not None else (base_cx, base_cy)
    elif resolved_mode == "skin_centroid":
        c = _centroid(slice_2d == 1)
        if c is None:
            c = _centroid(tissue_mask)
        cx, cy = c if c is not None else (base_cx, base_cy)
    elif resolved_mode == "ring_fit":
        max_shift_cells = int(round(center_search_m / dx))
        shifts = np.arange(-max_shift_cells, max_shift_cells + 1) * dx
        best = None

        for sx in shifts:
            for sy in shifts:
                cand_cx = base_cx + float(sx)
                cand_cy = base_cy + float(sy)
                positions = _antenna_positions_with_center(array_geo, cand_cx, cand_cy)
                score, n_oob, n_in_tissue, min_clear = _placement_score(
                    slice_2d, dx, positions, clearance_map, min_clearance_m
                )
                # Tie-break by minimal displacement from base center.
                tie = float(sx * sx + sy * sy)
                candidate = (score, tie, cand_cx, cand_cy, n_oob, n_in_tissue, min_clear)
                if best is None or candidate < best:
                    best = candidate

        if best is None:
            cx, cy = base_cx, base_cy
        else:
            _, _, cx, cy, _, _, _ = best
    else:
        raise ValueError(
            "Unknown center_mode: "
            f"{center_mode}. Expected one of: auto, volume, tissue_centroid, "
            "skin_centroid, ring_fit"
        )

    return {
        "cx": float(cx),
        "cy": float(cy),
        "base_cx": float(base_cx),
        "base_cy": float(base_cy),
        "slice_desc": slice_desc,
        "center_mode": center_mode,
        "resolved_mode": resolved_mode,
        "shift_x_mm": float((cx - base_cx) * 1000.0),
        "shift_y_mm": float((cy - base_cy) * 1000.0),
    }


def evaluate_array_placement(
    geo_info,
    array_geo,
    min_clearance_m=0.0,
    center_mode="auto",
    center_search_m=0.02,
    ant_z_m=None,
):
    """Evaluate antenna placement quality without generating input files.

    Returns placement statistics that can be used to audit geometry choices
    (radius/padding/centering mode) across phantoms.
    """
    dx = geo_info["dx"]
    if "labels_3d" in geo_info and ant_z_m is None:
        cz = geo_info["center_m"][2]
        ant_z_m = round(cz / dx) * dx

    center_info = _resolve_array_center(
        geo_info,
        array_geo,
        center_mode=center_mode,
        min_clearance_m=min_clearance_m,
        ant_z_m=ant_z_m,
        center_search_m=center_search_m,
    )
    cx, cy = center_info["cx"], center_info["cy"]
    positions = _antenna_positions_with_center(array_geo, cx, cy)
    slice_2d, slice_desc = _get_placement_slice(geo_info, ant_z_m=ant_z_m)

    ix = np.rint(positions[:, 0] / dx).astype(int)
    iy = np.rint(positions[:, 1] / dx).astype(int)
    in_bounds = (
        (ix >= 0) & (ix < slice_2d.shape[0]) &
        (iy >= 0) & (iy < slice_2d.shape[1])
    )

    n_ant = positions.shape[0]
    n_oob = int(np.sum(~in_bounds))

    tissue_mask = slice_2d > 0
    clearance_map = distance_transform_edt(~tissue_mask) * dx

    n_in_tissue = 0
    min_clearance_found = float("inf")
    if np.any(in_bounds):
        ix_ok = ix[in_bounds]
        iy_ok = iy[in_bounds]
        sampled_labels = slice_2d[ix_ok, iy_ok]
        n_in_tissue = int(np.sum(sampled_labels > 0))
        clearances = clearance_map[ix_ok, iy_ok]
        min_clearance_found = float(np.min(clearances))

    tissue_ix, tissue_iy = np.where(tissue_mask)
    if tissue_ix.size > 0:
        x_m = tissue_ix * dx
        y_m = tissue_iy * dx
        radii = np.sqrt((x_m - cx) ** 2 + (y_m - cy) ** 2)
        suggested_radius_m = float(np.max(radii) + min_clearance_m)
    else:
        suggested_radius_m = 0.0

    valid = (
        n_oob == 0
        and n_in_tissue == 0
        and min_clearance_found >= min_clearance_m
    )
    return {
        "valid": bool(valid),
        "n_antennas": int(n_ant),
        "n_oob": n_oob,
        "n_in_tissue": int(n_in_tissue),
        "min_clearance_m": float(min_clearance_found),
        "required_clearance_m": float(min_clearance_m),
        "suggested_radius_m": float(suggested_radius_m),
        "slice_desc": slice_desc,
        "array_center_m": (float(cx), float(cy)),
        "array_center_shift_mm": (
            float(center_info["shift_x_mm"]),
            float(center_info["shift_y_mm"]),
        ),
        "center_mode": center_info["center_mode"],
        "center_mode_resolved": center_info["resolved_mode"],
        "ant_z_m": None if ant_z_m is None else float(ant_z_m),
    }


def _check_antenna_positions(
    geo_info,
    positions_xy,
    min_clearance_m=0.0,
    ant_z_m=None,
):
    """Validate that antenna points are in coupling medium with clearance.

    Parameters
    ----------
    geo_info : dict
        Geometry info from prepare_2d_geometry or prepare_3d_geometry.
    positions_xy : ndarray
        Antenna (x, y) positions in meters, in domain coordinates.
    min_clearance_m : float
        Required minimum distance from each antenna point to tissue.
    ant_z_m : float or None
        z-position for 3D placement. Ignored for 2D geometry.

    Raises
    ------
    ValueError
        If any antenna is out of bounds, inside tissue, or too close.
    """
    dx = geo_info["dx"]
    center_x = float(np.mean(positions_xy[:, 0]))
    center_y = float(np.mean(positions_xy[:, 1]))
    n_ant = positions_xy.shape[0]
    slice_2d, slice_desc = _get_placement_slice(geo_info, ant_z_m=ant_z_m)

    ix = np.rint(positions_xy[:, 0] / dx).astype(int)
    iy = np.rint(positions_xy[:, 1] / dx).astype(int)
    in_bounds = (
        (ix >= 0) & (ix < slice_2d.shape[0]) &
        (iy >= 0) & (iy < slice_2d.shape[1])
    )

    n_oob = int(np.sum(~in_bounds))
    if n_oob > 0:
        raise ValueError(
            f"Invalid antenna placement ({slice_desc}): "
            f"{n_oob}/{n_ant} antennas are outside simulation domain."
        )

    sampled_labels = slice_2d[ix, iy]
    in_tissue = sampled_labels > 0
    n_in_tissue = int(np.sum(in_tissue))

    tissue_mask = slice_2d > 0
    clearance_map = distance_transform_edt(~tissue_mask) * dx
    clearances = clearance_map[ix, iy]
    min_clearance_found = float(np.min(clearances)) if clearances.size else float("inf")

    tissue_ix, tissue_iy = np.where(tissue_mask)
    if tissue_ix.size > 0:
        x_m = tissue_ix * dx
        y_m = tissue_iy * dx
        radii = np.sqrt((x_m - center_x) ** 2 + (y_m - center_y) ** 2)
        suggested_radius_m = float(np.max(radii) + min_clearance_m)
    else:
        suggested_radius_m = 0.0

    failing = n_in_tissue > 0 or min_clearance_found < min_clearance_m
    if failing:
        raise ValueError(
            "Invalid antenna placement "
            f"({slice_desc}): {n_in_tissue}/{n_ant} antennas are in tissue, "
            f"minimum clearance={min_clearance_found * 1000:.2f}mm "
            f"(required >= {min_clearance_m * 1000:.2f}mm). "
            f"Suggested ring radius from current center: >= {suggested_radius_m * 100:.2f}cm. "
            "Increase array_geo.radius_m and/or recenter the geometry."
        )


def generate_gprmax_inputs(
    geo_info,
    array_geo,
    work_dir,
    time_window=8e-9,
    validate_placement=True,
    min_clearance_m=0.0,
    center_mode="auto",
    center_search_m=0.02,
):
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
    validate_placement : bool
        If True, fail fast when antennas are out-of-domain / in tissue.
    min_clearance_m : float
        Required minimum distance from antennas to tissue (meters).
    center_mode : str
        Array-center strategy: auto, volume, tissue_centroid, skin_centroid, ring_fit.
    center_search_m : float
        Search half-width (meters) used by ring_fit.

    Returns
    -------
    list of (input_path, tx_idx, rx_idx)
    """
    dx = geo_info["dx"]
    domain_x, domain_y = geo_info["domain_m"]
    offset = geo_info["pad_cells"] * dx

    center_info = _resolve_array_center(
        geo_info,
        array_geo,
        center_mode=center_mode,
        min_clearance_m=min_clearance_m,
        center_search_m=center_search_m,
    )
    cx, cy = center_info["cx"], center_info["cy"]
    positions = _antenna_positions_with_center(array_geo, cx, cy)

    geo_info["array_center_m"] = (cx, cy)
    geo_info["array_center_shift_mm"] = (
        center_info["shift_x_mm"],
        center_info["shift_y_mm"],
    )
    geo_info["array_center_mode"] = center_info["center_mode"]
    geo_info["array_center_mode_resolved"] = center_info["resolved_mode"]
    if validate_placement:
        _check_antenna_positions(
            geo_info,
            positions,
            min_clearance_m=min_clearance_m,
        )

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


def generate_gprmax_inputs_3d(
    geo_info,
    array_geo,
    work_dir,
    time_window=12e-9,
    validate_placement=True,
    min_clearance_m=0.0,
    center_mode="auto",
    center_search_m=0.02,
):
    """Generate gprMax .in files for all Tx-Rx pairs in 3D.

    Antennas are placed in a ring at z = breast center.

    Parameters
    ----------
    geo_info : dict
        From prepare_3d_geometry().
    array_geo : ArrayGeometry
    work_dir : str
    time_window : float
    validate_placement : bool
        If True, fail fast when antennas are out-of-domain / in tissue.
    min_clearance_m : float
        Required minimum distance from antennas to tissue (meters).
    center_mode : str
        Array-center strategy: auto, volume, tissue_centroid, skin_centroid, ring_fit.
    center_search_m : float
        Search half-width (meters) used by ring_fit.

    Returns
    -------
    list of (input_path, tx_idx, rx_idx)
    """
    dx = geo_info["dx"]
    domain_x, domain_y, domain_z = geo_info["domain_m"]
    offset = geo_info["pad_cells"] * dx
    _, _, cz = geo_info["center_m"]
    ant_z = round(cz / dx) * dx  # snap to grid

    center_info = _resolve_array_center(
        geo_info,
        array_geo,
        center_mode=center_mode,
        min_clearance_m=min_clearance_m,
        ant_z_m=ant_z,
        center_search_m=center_search_m,
    )
    cx, cy = center_info["cx"], center_info["cy"]
    positions = _antenna_positions_with_center(array_geo, cx, cy)

    geo_info["array_center_m"] = (cx, cy)
    geo_info["array_center_shift_mm"] = (
        center_info["shift_x_mm"],
        center_info["shift_y_mm"],
    )
    geo_info["array_center_mode"] = center_info["center_mode"]
    geo_info["array_center_mode_resolved"] = center_info["resolved_mode"]
    if validate_placement:
        _check_antenna_positions(
            geo_info,
            positions,
            min_clearance_m=min_clearance_m,
            ant_z_m=ant_z,
        )

    pairs = array_geo.tx_rx_pairs()
    inputs = []

    for pair_idx, (tx_idx, rx_idx) in enumerate(pairs):
        tx_x = round(positions[tx_idx, 0] / dx) * dx
        tx_y = round(positions[tx_idx, 1] / dx) * dx
        rx_x = round(positions[rx_idx, 0] / dx) * dx
        rx_y = round(positions[rx_idx, 1] / dx) * dx

        lines = [
            f"#title: 3D scan pair {pair_idx} (tx={tx_idx} rx={rx_idx})",
            f"#domain: {domain_x:.6f} {domain_y:.6f} {domain_z:.6f}",
            f"#dx_dy_dz: {dx:.6f} {dx:.6f} {dx:.6f}",
            f"#time_window: {time_window:.2e}",
            "",
            "#material: 10.0 0.01 1 0 bg_coupling",
            f"#box: 0 0 0 {domain_x:.6f} {domain_y:.6f} {domain_z:.6f} bg_coupling",
            "",
            f"#geometry_objects_read: {offset:.6f} {offset:.6f} {offset:.6f} "
            "breast_geo.h5 breast_mat.txt",
            "",
            "#waveform: ricker 1 4.5e9 uwb_pulse",
            f"#hertzian_dipole: z {tx_x:.6f} {tx_y:.6f} {ant_z:.6f} uwb_pulse",
            "",
            f"#rx: {rx_x:.6f} {rx_y:.6f} {ant_z:.6f}",
        ]

        path = os.path.join(work_dir, f"pair_{pair_idx:04d}.in")
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        inputs.append((path, tx_idx, rx_idx))

    return inputs


def run_simulation(input_file, gpu=None):
    """Run a single gprMax simulation.

    Parameters
    ----------
    input_file : str
    gpu : list of int or None
        GPU device IDs (e.g. [0]). None = CPU.
    """
    from gprMax.gprMax import api
    if gpu is not None:
        api(input_file, gpu=gpu)
    else:
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
