"""
Visualize UWCEM phantom slices with and without tumor.
Saves PNG images for inspection.
"""

import sys
import numpy as np

# Check matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not installed. Install with: pip3 install matplotlib")
    sys.exit(1)

from wavecare.synth.phantoms import load_uwcem_phantom, UWCEM_TISSUE_MAP
from wavecare.synth.tumors import insert_tumor, random_tumor_position, random_tumor_params
from wavecare.synth.scenes import generate_scene

import os

# Resolve paths relative to project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.environ.get("WAVECARE_DATA_DIR",
                          os.path.join(_PROJECT_ROOT, "data", "uwcem"))
OUT_DIR = os.environ.get("WAVECARE_OUTPUT_DIR",
                         os.path.join(_PROJECT_ROOT, "output", "viz"))
os.makedirs(OUT_DIR, exist_ok=True)

# Color map for tissue types (0-10)
TISSUE_COLORS = [
    "#000000",  # 0: background (black)
    "#FFD700",  # 1: skin (gold)
    "#8B0000",  # 2: muscle (dark red)
    "#FFF8DC",  # 3: fat_1 (cornsilk)
    "#FAEBD7",  # 4: fat_2 (antique white)
    "#FFE4C4",  # 5: fat_3 (bisque)
    "#90EE90",  # 6: transitional (light green)
    "#4169E1",  # 7: fibro_1 (royal blue)
    "#1E90FF",  # 8: fibro_2 (dodger blue)
    "#00BFFF",  # 9: fibro_3 (deep sky blue)
    "#FF0000",  # 10: tumor (red)
]
cmap = ListedColormap(TISSUE_COLORS)


def visualize_phantom(phantom_id):
    phantom_dir = os.path.join(DATA_DIR, phantom_id)
    phantom = load_uwcem_phantom(phantom_dir)
    mtype = phantom["mtype"]
    info = phantom["info"]
    voxel_mm = phantom["voxel_mm"]

    # Generate scene with tumor
    rng = np.random.default_rng(42)
    scene = generate_scene(phantom, has_tumor=True, rng=rng,
                           tumor_params={"diameter_mm": 15.0, "shape": "sphere"})
    mtype_tumor = scene["mtype"]

    # Central slices
    mid_x = mtype.shape[0] // 2
    mid_y = mtype.shape[1] // 2
    mid_z = mtype.shape[2] // 2

    # If tumor was inserted, find the slice that passes through it
    if scene["tumor_info"]:
        tc = scene["tumor_info"]["center_mm"]
        tumor_z = int(tc[2] / voxel_mm)
        tumor_z = min(max(tumor_z, 0), mtype.shape[2] - 1)
    else:
        tumor_z = mid_z

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"UWCEM Phantom {phantom_id} (ACR Class {info.get('classification', '?')})",
                 fontsize=16)

    # Row 1: original phantom
    titles_orig = [
        f"Axial (z={mid_z})",
        f"Coronal (y={mid_y})",
        f"Sagittal (x={mid_x})",
    ]
    slices_orig = [
        mtype[:, :, mid_z],
        mtype[:, mid_y, :],
        mtype[mid_x, :, :],
    ]

    for ax, sl, title in zip(axes[0], slices_orig, titles_orig):
        im = ax.imshow(sl.T, cmap=cmap, vmin=0, vmax=10,
                       origin='lower', interpolation='nearest')
        ax.set_title(f"Original - {title}")
        ax.set_xlabel("mm" if voxel_mm == 0.5 else "voxels")

    # Row 2: with tumor
    titles_tumor = [
        f"Axial (z={tumor_z}) - with tumor",
        f"Coronal (y={mid_y}) - with tumor",
        f"Sagittal (x={mid_x}) - with tumor",
    ]
    slices_tumor = [
        mtype_tumor[:, :, tumor_z],
        mtype_tumor[:, mid_y, :],
        mtype_tumor[mid_x, :, :],
    ]

    for ax, sl, title in zip(axes[1], slices_tumor, titles_tumor):
        im = ax.imshow(sl.T, cmap=cmap, vmin=0, vmax=10,
                       origin='lower', interpolation='nearest')
        ax.set_title(title)

    # Legend
    tissue_names = {v: k for k, v in UWCEM_TISSUE_MAP.items()}
    tissue_names[10] = "tumor"
    handles = []
    for i, (color, name) in enumerate(zip(TISSUE_COLORS,
            ["bg", "skin", "muscle", "fat1", "fat2", "fat3",
             "trans", "fib1", "fib2", "fib3", "TUMOR"])):
        handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=name))
    fig.legend(handles=handles, loc='lower center', ncol=11, fontsize=10)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = os.path.join(OUT_DIR, f"phantom_{phantom_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

    if scene["tumor_info"]:
        ti = scene["tumor_info"]
        print(f"  Tumor: {ti['diameter_mm']:.1f}mm sphere at "
              f"({ti['tum_x']:.2f}, {ti['tum_y']:.2f}, {ti['tum_z']:.2f}) cm")
        print(f"  Tumor voxels: {ti['n_voxels']}")


if __name__ == "__main__":
    for pid in ["071904", "062204"]:
        if os.path.exists(os.path.join(DATA_DIR, pid)):
            visualize_phantom(pid)
    print("\nDone! Check output/viz/ for PNG images.")
