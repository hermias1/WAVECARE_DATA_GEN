"""
Scene generator: assembles phantoms + tumors + dielectric randomization.

A "scene" is a fully specified 3D breast model ready for EM simulation,
including tissue geometry, dielectric properties, and metadata.
"""

import numpy as np
from .tumors import insert_tumor, random_tumor_position, random_tumor_params


# UMBMID-like BIRADS classification based on fibroglandular fraction
BIRADS_THRESHOLDS = {
    1: (0.0, 0.25),    # mostly fatty
    2: (0.25, 0.50),   # scattered
    3: (0.50, 0.75),   # heterogeneously dense
    4: (0.75, 1.0),    # extremely dense
}


def compute_birads(mtype, fibro_labels=None):
    """Estimate ACR BIRADS class from tissue label volume.

    Parameters
    ----------
    mtype : ndarray
        3D tissue label array.
    fibro_labels : set or None
        Label values considered fibroglandular. If None, uses {6,7,8,9}
        (UWCEM transitional + fibroglandular).

    Returns
    -------
    birads : int (1-4)
    fib_fraction : float
    """
    if fibro_labels is None:
        fibro_labels = {6, 7, 8, 9}

    # Interior = everything that's not air (0) or skin (1)
    interior = (mtype > 1)
    n_interior = np.sum(interior)
    if n_interior == 0:
        return 1, 0.0

    n_fibro = sum(np.sum(mtype == label) for label in fibro_labels)
    fib_fraction = n_fibro / n_interior

    for birads_class, (lo, hi) in BIRADS_THRESHOLDS.items():
        if lo <= fib_fraction < hi:
            return birads_class, fib_fraction

    return 4, fib_fraction


def generate_scene(phantom, has_tumor=True, tumor_params=None, rng=None,
                   dielectric_perturbation=0.1, tumor_label=10):
    """Generate a complete scene for EM simulation.

    Parameters
    ----------
    phantom : dict
        Output from phantoms.load_uwcem_phantom() or load_pelicano_phantom().
    has_tumor : bool
        Whether to insert a tumor.
    tumor_params : dict or None
        If None, generates random tumor parameters.
    rng : numpy.random.Generator or None
    dielectric_perturbation : float
        Relative std for randomizing dielectric properties (0 = no noise).
    tumor_label : int
        Integer label for tumor voxels.

    Returns
    -------
    scene : dict
        'mtype': 3D ndarray with (possibly) tumor inserted,
        'voxel_mm': float,
        'has_tumor': bool,
        'tumor_info': dict or None (center_mm, diameter_mm, type, ...),
        'birads': int,
        'fib_fraction': float,
        'dielectric_perturbation': float,
        'seed': int (for reproducibility)
    """
    if rng is None:
        rng = np.random.default_rng()

    seed = int(rng.integers(0, 2**31))
    mtype = phantom["mtype"].copy()
    voxel_mm = phantom["voxel_mm"]

    tumor_info = None

    if has_tumor:
        if tumor_params is None:
            tumor_params = random_tumor_params(rng=rng)

        # Find a valid position
        center_mm = random_tumor_position(mtype, voxel_mm, rng=rng)
        tumor_params["center_mm"] = center_mm

        mtype, n_voxels = insert_tumor(
            mtype, voxel_mm, tumor_params, tumor_label=tumor_label, rng=rng
        )

        # Convert center from voxel-origin mm to UMBMID-like coordinates
        # (centered on breast, in cm)
        shape_mm = np.array(mtype.shape) * voxel_mm
        center_cm = (np.array(center_mm) - shape_mm / 2) / 10.0

        diam = tumor_params["diameter_mm"]
        if isinstance(diam, (list, tuple)):
            # For ellipsoids, report equivalent diameter
            equiv_diam_mm = (diam[0] * diam[1] * diam[2]) ** (1/3)
        else:
            equiv_diam_mm = diam

        tumor_info = {
            "center_mm": center_mm,
            "center_cm": tuple(center_cm),
            "tum_x": center_cm[0],
            "tum_y": center_cm[1],
            "tum_z": center_cm[2],
            "diameter_mm": diam,
            "tum_diam": equiv_diam_mm / 10.0,  # in cm for UMBMID compat
            "tum_rad": equiv_diam_mm / 20.0,   # radius in cm
            "shape": tumor_params["shape"],
            "type": tumor_params.get("type", "malignant"),
            "n_voxels": n_voxels,
        }

    birads, fib_fraction = compute_birads(mtype)

    return {
        "mtype": mtype,
        "voxel_mm": voxel_mm,
        "has_tumor": has_tumor,
        "tumor_info": tumor_info,
        "birads": birads,
        "fib_fraction": fib_fraction,
        "dielectric_perturbation": dielectric_perturbation,
        "seed": seed,
    }


def scene_to_metadata(scene, scan_id=0, phant_id="SYN001"):
    """Convert scene dict to UMBMID-compatible metadata dict.

    Parameters
    ----------
    scene : dict
        Output from generate_scene().
    scan_id : int
        Unique scan identifier.
    phant_id : str
        Phantom identifier string.

    Returns
    -------
    metadata : dict
        UMBMID-compatible metadata dictionary.
    """
    md = {
        "n_expt": scan_id,
        "id": scan_id,
        "phant_id": phant_id,
        "birads": scene["birads"],
        "source": "synthetic",
    }

    if scene["has_tumor"] and scene["tumor_info"] is not None:
        ti = scene["tumor_info"]
        md["tum_rad"] = ti["tum_rad"]
        md["tum_x"] = ti["tum_x"]
        md["tum_y"] = ti["tum_y"]
        md["tum_z"] = ti["tum_z"]
        md["tum_diam"] = ti["tum_diam"]
        md["tum_shape"] = ti["shape"]
    else:
        md["tum_rad"] = float('nan')
        md["tum_x"] = float('nan')
        md["tum_y"] = float('nan')
        md["tum_z"] = float('nan')
        md["tum_diam"] = float('nan')
        md["tum_shape"] = ""

    return md
