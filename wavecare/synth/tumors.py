"""
Tumor insertion into 3D breast phantom volumes.

Handles placement of spherical and ellipsoidal tumors at randomized
positions within the breast volume, respecting anatomical constraints.
"""

import numpy as np


def insert_tumor(mtype, voxel_mm, tumor_params, tumor_label=10, rng=None):
    """Insert a tumor into a 3D tissue label volume.

    Parameters
    ----------
    mtype : ndarray (int)
        3D array of tissue labels. Modified in-place.
    voxel_mm : float
        Voxel size in mm.
    tumor_params : dict
        'center_mm': (x, y, z) in mm from volume origin,
        'diameter_mm': float (for sphere) or (dx, dy, dz) for ellipsoid,
        'shape': 'sphere' or 'ellipsoid'
    tumor_label : int
        Label value to assign to tumor voxels.
    rng : numpy.random.Generator or None

    Returns
    -------
    mtype : ndarray
        Modified volume with tumor inserted.
    n_voxels : int
        Number of voxels set to tumor.
    """
    cx, cy, cz = np.array(tumor_params["center_mm"]) / voxel_mm
    shape_type = tumor_params.get("shape", "sphere")

    if shape_type == "sphere":
        diam = tumor_params["diameter_mm"]
        rx = ry = rz = (diam / 2.0) / voxel_mm
    else:  # ellipsoid
        dx, dy, dz = tumor_params["diameter_mm"]
        rx, ry, rz = (dx / 2.0) / voxel_mm, (dy / 2.0) / voxel_mm, (dz / 2.0) / voxel_mm

    nx, ny, nz = mtype.shape

    # Bounding box (clipped to volume)
    x0 = max(int(cx - rx) - 1, 0)
    x1 = min(int(cx + rx) + 2, nx)
    y0 = max(int(cy - ry) - 1, 0)
    y1 = min(int(cy + ry) + 2, ny)
    z0 = max(int(cz - rz) - 1, 0)
    z1 = min(int(cz + rz) + 2, nz)

    # Create coordinate grids in the bounding box
    xx, yy, zz = np.mgrid[x0:x1, y0:y1, z0:z1]

    # Ellipsoid equation: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 + ((z-cz)/rz)^2 <= 1
    dist_sq = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + ((zz - cz) / rz) ** 2
    mask = dist_sq <= 1.0

    # Only place tumor in non-background, non-skin voxels
    # (i.e., inside the breast, not in air or skin)
    tissue_ok = (mtype[x0:x1, y0:y1, z0:z1] > 1)
    final_mask = mask & tissue_ok

    mtype[x0:x1, y0:y1, z0:z1][final_mask] = tumor_label
    n_voxels = int(np.sum(final_mask))

    return mtype, n_voxels


def random_tumor_position(mtype, voxel_mm, margin_mm=5.0, rng=None):
    """Pick a random position inside the breast volume.

    Avoids skin (label 1) and air (label 0), and stays margin_mm
    away from the breast surface.

    Parameters
    ----------
    mtype : ndarray (int)
        3D tissue label volume.
    voxel_mm : float
        Voxel size in mm.
    margin_mm : float
        Minimum distance from skin/air boundary.
    rng : numpy.random.Generator or None

    Returns
    -------
    center_mm : tuple (x, y, z)
        Tumor center in mm from volume origin.
    """
    if rng is None:
        rng = np.random.default_rng()

    margin_vox = int(margin_mm / voxel_mm)

    # Find interior voxels (not air=0, not skin=1)
    interior = (mtype > 1)

    # Erode by margin to stay away from surface
    if margin_vox > 0:
        from scipy.ndimage import binary_erosion
        struct = np.ones((2 * margin_vox + 1,) * 3)
        interior = binary_erosion(interior, structure=struct)

    candidates = np.argwhere(interior)
    if len(candidates) == 0:
        raise ValueError("No valid interior voxels for tumor placement")

    idx = rng.integers(0, len(candidates))
    voxel_pos = candidates[idx]
    center_mm = tuple(voxel_pos * voxel_mm)

    return center_mm


def random_tumor_params(rng=None, diameter_range_mm=(5.0, 30.0),
                        ellipsoid_ratio_range=(1.0, 2.0),
                        p_ellipsoid=0.3):
    """Generate random tumor parameters.

    Parameters
    ----------
    rng : numpy.random.Generator or None
    diameter_range_mm : tuple
        (min, max) diameter in mm.
    ellipsoid_ratio_range : tuple
        (min, max) aspect ratio for ellipsoids.
    p_ellipsoid : float
        Probability of generating an ellipsoid vs sphere.

    Returns
    -------
    params : dict
        'diameter_mm': float or tuple,
        'shape': 'sphere' or 'ellipsoid',
        'type': 'malignant' or 'benign'
    """
    if rng is None:
        rng = np.random.default_rng()

    diam = rng.uniform(*diameter_range_mm)

    if rng.random() < p_ellipsoid:
        ratio = rng.uniform(*ellipsoid_ratio_range)
        # Random axis to elongate
        axis = rng.integers(0, 3)
        diams = [diam, diam, diam]
        diams[axis] *= ratio
        shape_type = "ellipsoid"
        diameter = tuple(diams)
    else:
        shape_type = "sphere"
        diameter = diam

    tumor_type = "malignant" if rng.random() < 0.7 else "benign"

    return {
        "diameter_mm": diameter,
        "shape": shape_type,
        "type": tumor_type,
    }
