"""
Test UWCEM phantom loading and basic scene generation.
"""

import os
import numpy as np

from wavecare.synth.phantoms import load_uwcem_phantom, UWCEM_TISSUE_MAP

DATA_DIR = "/Users/ulysse/Desktop/ML/Wavecare/data/uwcem"

def test_load_phantom(phantom_id):
    """Load a phantom and print summary statistics."""
    phantom_dir = os.path.join(DATA_DIR, phantom_id)
    print(f"\n{'='*60}")
    print(f"Loading phantom: {phantom_id}")
    print(f"{'='*60}")

    phantom = load_uwcem_phantom(phantom_dir)

    mtype = phantom["mtype"]
    info = phantom["info"]
    voxel_mm = phantom["voxel_mm"]

    print(f"  Shape: {mtype.shape}")
    print(f"  Voxel size: {voxel_mm} mm")
    print(f"  Physical size: {mtype.shape[0]*voxel_mm/10:.1f} x "
          f"{mtype.shape[1]*voxel_mm/10:.1f} x "
          f"{mtype.shape[2]*voxel_mm/10:.1f} cm")
    print(f"  Total voxels: {mtype.size:,}")
    print(f"  ACR class: {info.get('classification', '?')}")
    print(f"  dtype: {mtype.dtype}")
    print(f"  Unique labels: {np.unique(mtype)}")

    # Tissue distribution
    print(f"\n  Tissue distribution:")
    total_nonair = np.sum(mtype > 0)
    for label in sorted(np.unique(mtype)):
        count = np.sum(mtype == label)
        pct = 100 * count / mtype.size
        name = UWCEM_TISSUE_MAP.get(label, f"unknown({label})")
        marker = " <--" if label > 0 and count == max(
            np.sum(mtype == l) for l in range(1, 10)
        ) else ""
        print(f"    {label} ({name:20s}): {count:>10,} voxels ({pct:5.1f}%){marker}")

    # Compute fibroglandular fraction (for BIRADS)
    interior = (mtype > 1)  # exclude air and skin
    n_interior = np.sum(interior)
    fibro_labels = {6, 7, 8, 9}  # transitional + fibroglandular
    n_fibro = sum(np.sum(mtype == l) for l in fibro_labels)
    fib_frac = n_fibro / n_interior if n_interior > 0 else 0
    print(f"\n  Fibroglandular fraction: {fib_frac:.3f} ({fib_frac*100:.1f}%)")

    # Central slice visualization info
    mid_z = mtype.shape[2] // 2
    slice_2d = mtype[:, :, mid_z]
    print(f"\n  Central slice (z={mid_z}): {slice_2d.shape}")
    print(f"  Tissues in slice: {np.unique(slice_2d)}")

    return phantom


def test_tumor_insertion(phantom):
    """Test inserting a tumor into the phantom."""
    from wavecare.synth.tumors import insert_tumor, random_tumor_position, random_tumor_params

    mtype = phantom["mtype"].copy()
    voxel_mm = phantom["voxel_mm"]
    rng = np.random.default_rng(42)

    print(f"\n  --- Tumor insertion test ---")

    # Generate random tumor
    params = random_tumor_params(rng=rng, diameter_range_mm=(10.0, 20.0))
    print(f"  Tumor params: {params}")

    # Find position
    try:
        center = random_tumor_position(mtype, voxel_mm, margin_mm=5.0, rng=rng)
        print(f"  Position: {center} mm")
        params["center_mm"] = center

        # Insert
        mtype_with_tumor, n_vox = insert_tumor(mtype, voxel_mm, params,
                                                tumor_label=10, rng=rng)
        print(f"  Tumor voxels inserted: {n_vox}")
        print(f"  Unique labels after: {np.unique(mtype_with_tumor)}")

        # Verify tumor is there
        assert 10 in mtype_with_tumor, "Tumor label 10 not found!"
        print("  PASS: Tumor successfully inserted")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    phantoms = {}

    for pid in ["071904", "062204"]:
        phantom_dir = os.path.join(DATA_DIR, pid)
        if os.path.exists(phantom_dir):
            phantom = test_load_phantom(pid)
            phantoms[pid] = phantom
            test_tumor_insertion(phantom)
        else:
            print(f"Phantom {pid} not found at {phantom_dir}")

    print(f"\n{'='*60}")
    print("All tests done!")
