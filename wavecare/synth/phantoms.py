"""
Phantom loaders for UWCEM and Pelicano 2024 breast models.

UWCEM: 11 MRI-derived phantoms at 0.5mm resolution, 10 tissue types.
Pelicano: 55 patient-derived models at ~1mm resolution, with tumors.

Both provide 3D voxel grids with tissue labels and dielectric assignments.
"""

import os
import struct
import numpy as np

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


# --------------------------------------------------------------------------- #
# UWCEM phantom loader
# --------------------------------------------------------------------------- #

# UWCEM tissue label mapping: raw float values -> tissue names
# From the actual mtype.txt files in the UWCEM repository.
# Encoding: negative = structural tissues, 1.x = fatty, 2.x = transitional,
#           3.x = fibroglandular
UWCEM_RAW_TISSUE_MAP = {
    -4.0: "muscle",             # chest wall
    -2.0: "skin",               # 1.5mm skin layer
    -1.0: "background",         # immersion medium / air
     1.1: "fat_1",              # fatty type 1 (highest lipid)
     1.2: "fat_2",              # fatty type 2
     1.3: "fat_3",              # fatty type 3 (lowest lipid)
     2.0: "transitional",       # transitional tissue
     3.1: "fibro_1",            # fibroglandular type 1
     3.2: "fibro_2",            # fibroglandular type 2
     3.3: "fibro_3",            # fibroglandular type 3 (highest water)
}

# Integer label mapping (after conversion from raw floats)
UWCEM_TISSUE_MAP = {
    0: "background",       # air / immersion (-1.0)
    1: "skin",             # (-2.0)
    2: "muscle",           # (-4.0)
    3: "fat_1",            # (1.1)
    4: "fat_2",            # (1.2)
    5: "fat_3",            # (1.3)
    6: "transitional",     # (2.0)
    7: "fibro_1",          # (3.1)
    8: "fibro_2",          # (3.2)
    9: "fibro_3",          # (3.3)
}

# Map float values to integer labels
_UWCEM_FLOAT_TO_INT = {
    -1.0: 0,   # background
    -2.0: 1,   # skin
    -4.0: 2,   # muscle
     1.1: 3,   # fat_1
     1.2: 4,   # fat_2
     1.3: 5,   # fat_3
     2.0: 6,   # transitional
     3.1: 7,   # fibro_1
     3.2: 8,   # fibro_2
     3.3: 9,   # fibro_3
}

# Map integer labels to Lazebnik/Gabriel tissue names for dielectrics
UWCEM_TO_DIELECTRIC = {
    0: None,                    # background = air (eps_r=1)
    1: "skin_dry",
    2: "muscle",
    3: "adipose_low",
    4: "adipose_med",
    5: "adipose_high",
    6: "transitional_med",
    7: "fibroglandular_low",
    8: "fibroglandular_med",
    9: "fibroglandular_high",
}


def load_uwcem_phantom(phantom_dir):
    """Load a UWCEM phantom from its directory.

    Expects files: mtype.zip (containing mtype.txt), breastInfo.txt.
    The mtype.txt is a text file with one float value per line, using
    the UWCEM encoding (e.g., -1.0=background, 1.2=fat, 3.1=fibro).
    Values are converted to integer labels 0-9.

    Parameters
    ----------
    phantom_dir : str
        Path to the phantom directory (e.g., .../071904/).

    Returns
    -------
    phantom : dict
        'mtype': 3D ndarray (int8) of integer tissue labels (0-9),
        'mtype_raw': 3D ndarray (float32) of original float labels,
        'shape': (nx, ny, nz),
        'voxel_mm': voxel size in mm,
        'info': dict with metadata from breastInfo.txt
    """
    info = _parse_breast_info(os.path.join(phantom_dir, "breastInfo.txt"))
    nx, ny, nz = info["nx"], info["ny"], info["nz"]
    expected_size = nx * ny * nz

    # Load raw data
    raw_flat = _load_mtype_data(phantom_dir, expected_size)

    # Reshape: UWCEM uses MATLAB column-major (Fortran) order
    raw_3d = raw_flat.reshape((nx, ny, nz), order='F')

    # Convert float labels to integer labels
    mtype = _float_to_int_labels(raw_3d)

    return {
        "mtype": mtype,
        "mtype_raw": raw_3d,
        "shape": (nx, ny, nz),
        "voxel_mm": info.get("voxel_mm", 0.5),
        "info": info,
        "tissue_map": UWCEM_TISSUE_MAP,
        "raw_tissue_map": UWCEM_RAW_TISSUE_MAP,
    }


def _load_mtype_data(phantom_dir, expected_size):
    """Load mtype data from zip (text) or binary file."""
    import zipfile

    zip_path = os.path.join(phantom_dir, "mtype.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            data_name = names[0]  # typically mtype.txt
            raw_bytes = zf.read(data_name)

            if data_name.endswith('.txt'):
                # Text format: one float per line
                lines = raw_bytes.decode().strip().split('\n')
                raw_flat = np.array([float(l) for l in lines],
                                    dtype=np.float32)
            else:
                # Binary format
                raw_flat = np.frombuffer(raw_bytes, dtype=np.float32)

        if raw_flat.size != expected_size:
            raise ValueError(
                f"Data size mismatch: got {raw_flat.size}, "
                f"expected {expected_size}"
            )
        return raw_flat

    raise FileNotFoundError(
        f"No mtype data found in {phantom_dir}. "
        "Download from https://uwcem.ece.wisc.edu/phantomRepository.html"
    )


def _float_to_int_labels(raw_3d):
    """Convert UWCEM float tissue labels to integer labels 0-9."""
    mtype = np.zeros_like(raw_3d, dtype=np.int8)

    for float_val, int_val in _UWCEM_FLOAT_TO_INT.items():
        # Use approximate comparison for floats
        mask = np.abs(raw_3d - float_val) < 0.05
        mtype[mask] = int_val

    return mtype


def _parse_breast_info(path):
    """Parse breastInfo.txt for phantom dimensions and metadata.

    Format example:
        breast ID=071904
        s1=310
        s2=355
        s3=253
        classification=1

    s1, s2, s3 are the voxel grid dimensions (nx, ny, nz).
    """
    info = {}
    if not os.path.exists(path):
        return {"nx": 0, "ny": 0, "nz": 0, "voxel_mm": 0.5}

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip().lower().replace(' ', '_')
                val = val.strip().rstrip(';')
                try:
                    info[key] = int(val)
                except ValueError:
                    try:
                        info[key] = float(val)
                    except ValueError:
                        info[key] = val

    # Map s1/s2/s3 to nx/ny/nz
    if "s1" in info:
        info["nx"] = info["s1"]
    if "s2" in info:
        info["ny"] = info["s2"]
    if "s3" in info:
        info["nz"] = info["s3"]

    # Default voxel size (UWCEM standard = 0.5mm)
    if "voxel_mm" not in info:
        info["voxel_mm"] = 0.5

    return info


# --------------------------------------------------------------------------- #
# Pelicano 2024 phantom loader (MHA format via SimpleITK)
# --------------------------------------------------------------------------- #

# Pelicano label map (simple version)
PELICANO_LABEL_MAP = {
    -4: "tumor_benign",
    -3: "tumor_malignant",
    -2: "skin",
    -1: "muscle",
    0: "background",
    1: "fat",
}

PELICANO_TO_DIELECTRIC = {
    -4: "tumor_benign",
    -3: "tumor_malignant",
    -2: "skin_dry",
    -1: "muscle",
    0: None,
    1: "adipose_med",
}


def load_pelicano_phantom(patient_dir, label_type="simple"):
    """Load a Pelicano 2024 patient-derived breast model.

    Parameters
    ----------
    patient_dir : str
        Path to a patient directory containing Label_map_simple.mha
        (or Label_map_detailed.mha).
    label_type : str
        "simple" or "detailed".

    Returns
    -------
    phantom : dict
        'labels': 3D ndarray of tissue labels,
        'shape': (nx, ny, nz),
        'spacing_mm': tuple of voxel sizes,
        'origin': tuple,
        'direction': tuple
    """
    if not HAS_SITK:
        raise ImportError(
            "SimpleITK required for Pelicano phantoms: pip install SimpleITK"
        )

    label_file = f"Label_map_{label_type}.mha"
    path = os.path.join(patient_dir, label_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")

    img = sitk.ReadImage(path)
    labels = sitk.GetArrayFromImage(img)  # (z, y, x) convention in SimpleITK

    return {
        "labels": labels,
        "shape": labels.shape,
        "spacing_mm": img.GetSpacing(),
        "origin": img.GetOrigin(),
        "direction": img.GetDirection(),
        "label_map": PELICANO_LABEL_MAP if label_type == "simple"
                     else None,
    }


# --------------------------------------------------------------------------- #
# Phantom inventory
# --------------------------------------------------------------------------- #

def list_uwcem_phantoms(data_dir):
    """List available UWCEM phantom directories."""
    if not os.path.isdir(data_dir):
        return []
    return sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])


def list_pelicano_patients(data_dir):
    """List available Pelicano patient directories."""
    if not os.path.isdir(data_dir):
        return []
    return sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
