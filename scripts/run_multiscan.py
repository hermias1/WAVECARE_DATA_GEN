"""
Multi-angle scan on UWCEM phantom 071904 using gprMax.

2D axial slice through tumor center, 72 angles, Debye dispersive materials.
Matches UMBMID Gen-2 bistatic geometry (+60 deg Tx-Rx offset).
"""

import os
import time
import numpy as np
import h5py

from wavecare.synth.phantoms import load_uwcem_phantom
from wavecare.synth.scenes import generate_scene

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.environ.get("WAVECARE_DATA_DIR",
                          os.path.join(_PROJECT_ROOT, "data", "uwcem"))
WORK_DIR = os.path.join(_PROJECT_ROOT, "output", "multiscan")
OUT_DIR = os.path.join(_PROJECT_ROOT, "output", "viz")

PHANTOM_ID = "071904"
SLICE_IDX = 177  # axial slice through tumor center

N_ANGLES = 72
ANGLE_STEP_DEG = 5.0
START_ANGLE_DEG = -130.0
ANTENNA_OFFSET_DEG = 60.0
ANTENNA_RADIUS_M = 0.075

DX = 0.001       # 1mm cells
TIME_WINDOW = 8e-9
PAD_CELLS = 40    # 4cm padding each side

TUMOR_DIAMETER_MM = 12.0
RNG_SEED = 42


def prepare_geometry():
    """Load phantom, insert tumor, prepare 2D geometry files."""
    phantom = load_uwcem_phantom(os.path.join(DATA_DIR, PHANTOM_ID))
    rng = np.random.default_rng(RNG_SEED)
    scene = generate_scene(phantom, has_tumor=True, rng=rng,
                           tumor_params={"diameter_mm": TUMOR_DIAMETER_MM,
                                         "shape": "sphere"})
    mtype = scene["mtype"]

    # Axial slice at tumor center
    mtype_2d = mtype[:, :, SLICE_IDX].copy()

    # Resample 0.5mm -> 1mm
    scale = phantom["voxel_mm"] / (DX * 1000)
    new_nx = int(mtype_2d.shape[0] * scale)
    new_ny = int(mtype_2d.shape[1] * scale)
    x_idx = (np.arange(new_nx) / scale).astype(int).clip(0, mtype_2d.shape[0]-1)
    y_idx = (np.arange(new_ny) / scale).astype(int).clip(0, mtype_2d.shape[1]-1)
    mtype_2d = mtype_2d[np.ix_(x_idx, y_idx)]

    # Pad with coupling medium
    nx, ny = mtype_2d.shape
    padded = np.zeros((nx + 2*PAD_CELLS, ny + 2*PAD_CELLS), dtype=np.int8)
    padded[PAD_CELLS:PAD_CELLS+nx, PAD_CELLS:PAD_CELLS+ny] = mtype_2d

    domain_x = padded.shape[0] * DX
    domain_y = padded.shape[1] * DX
    center_x = (PAD_CELLS + nx/2) * DX
    center_y = (PAD_CELLS + ny/2) * DX

    # Write HDF5 geometry
    geo_file = os.path.join(WORK_DIR, "breast_geo.h5")
    with h5py.File(geo_file, 'w') as f:
        f.attrs['dx_dy_dz'] = (DX, DX, DX)
        f.create_dataset('data', data=padded[:, :, np.newaxis].astype(np.int16),
                         dtype='int16')

    # Write materials file
    mat_file = os.path.join(WORK_DIR, "breast_mat.txt")
    mat_lines = [
        "#material: 10.0 0.01 1 0 coupling_medium",
        "#material: 4.0 0.0002 1 0 skin",
        "#add_dispersion_debye: 1 32.0 7.234e-12 skin",
        "#material: 4.0 0.20 1 0 muscle",
        "#add_dispersion_debye: 1 50.0 7.234e-12 muscle",
        "#material: 2.55 0.020 1 0 fat_1",
        "#add_dispersion_debye: 1 1.20 13.0e-12 fat_1",
        "#material: 2.55 0.036 1 0 fat_2",
        "#add_dispersion_debye: 1 1.71 13.0e-12 fat_2",
        "#material: 3.0 0.083 1 0 fat_3",
        "#add_dispersion_debye: 1 3.65 13.0e-12 fat_3",
        "#material: 5.5 0.304 1 0 transitional",
        "#add_dispersion_debye: 1 16.55 13.0e-12 transitional",
        "#material: 9.9 0.462 1 0 fibro_1",
        "#add_dispersion_debye: 1 26.60 13.0e-12 fibro_1",
        "#material: 13.81 0.738 1 0 fibro_2",
        "#add_dispersion_debye: 1 35.55 13.0e-12 fibro_2",
        "#material: 6.15 0.809 1 0 fibro_3",
        "#add_dispersion_debye: 1 48.26 13.0e-12 fibro_3",
        "#material: 14.0 0.90 1 0 tumor",
        "#add_dispersion_debye: 1 42.0 13.0e-12 tumor",
    ]
    with open(mat_file, 'w') as f:
        f.write('\n'.join(mat_lines) + '\n')

    return (domain_x, domain_y), (center_x, center_y), scene["tumor_info"]


def generate_input_files(domain_size, center):
    """Generate .in files for all 72 angles."""
    domain_x, domain_y = domain_size
    center_x, center_y = center
    offset = PAD_CELLS * DX

    paths = []
    for i in range(N_ANGLES):
        angle_tx = np.deg2rad(START_ANGLE_DEG + i * ANGLE_STEP_DEG)
        tx_x = round((center_x + ANTENNA_RADIUS_M * np.cos(angle_tx)) / DX) * DX
        tx_y = round((center_y + ANTENNA_RADIUS_M * np.sin(angle_tx)) / DX) * DX

        angle_rx = angle_tx + np.deg2rad(ANTENNA_OFFSET_DEG)
        rx_x = round((center_x + ANTENNA_RADIUS_M * np.cos(angle_rx)) / DX) * DX
        rx_y = round((center_y + ANTENNA_RADIUS_M * np.sin(angle_rx)) / DX) * DX

        lines = [
            f"#title: UWCEM {PHANTOM_ID} angle {i}",
            f"#domain: {domain_x:.6f} {domain_y:.6f} {DX:.6f}",
            f"#dx_dy_dz: {DX:.6f} {DX:.6f} {DX:.6f}",
            f"#time_window: {TIME_WINDOW:.2e}",
            "",
            "#material: 10.0 0.01 1 0 bg_coupling",
            f"#box: 0 0 0 {domain_x:.6f} {domain_y:.6f} {DX:.6f} bg_coupling",
            "",
            f"#geometry_objects_read: {offset:.6f} {offset:.6f} 0 "
            "breast_geo.h5 breast_mat.txt",
            "",
            "#waveform: ricker 1 4.5e9 uwb_pulse",
            f"#hertzian_dipole: z {tx_x:.6f} {tx_y:.6f} 0 uwb_pulse",
            "",
            f"#rx: {rx_x:.6f} {rx_y:.6f} 0",
        ]

        path = os.path.join(WORK_DIR, f"angle_{i:02d}.in")
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        paths.append(path)

    return paths


def run_all(input_files):
    """Run all simulations sequentially."""
    from gprMax.gprMax import api

    t0 = time.time()
    for i, infile in enumerate(input_files):
        ts = time.time()
        try:
            api(infile)
        except Exception as e:
            print(f"  FAILED angle {i}: {e}")
            continue

        elapsed = time.time() - ts
        if i == 0:
            est = elapsed * N_ANGLES / 60
            print(f"\n  Angle 0: {elapsed:.1f}s (est. total: {est:.1f} min)")
        elif (i + 1) % 12 == 0:
            total = time.time() - t0
            eta = total / (i + 1) * (N_ANGLES - i - 1) / 60
            print(f"  [{i+1}/{N_ANGLES}] {elapsed:.1f}s/sim, ETA: {eta:.1f} min")

    total = time.time() - t0
    print(f"\n  Total: {total/60:.1f} minutes ({total/N_ANGLES:.1f}s/sim avg)")


def collect_results():
    """Collect all .out files into single array."""
    all_ez = []
    dt = None

    for i in range(N_ANGLES):
        out = os.path.join(WORK_DIR, f"angle_{i:02d}.out")
        if not os.path.exists(out):
            all_ez.append(None)
            continue
        with h5py.File(out, 'r') as f:
            ez = f['rxs']['rx1']['Ez'][:]
            if dt is None:
                dt = f.attrs['dt']
            all_ez.append(ez)

    max_len = max(len(e) for e in all_ez if e is not None)
    data = np.zeros((N_ANGLES, max_len))
    valid = 0
    for i, ez in enumerate(all_ez):
        if ez is not None:
            data[i, :len(ez)] = ez
            valid += 1

    print(f"  Collected {valid}/{N_ANGLES} traces, shape: {data.shape}")
    return data, dt


def visualize(data, dt, tumor_info):
    """Generate sinogram and analysis plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.fft import fft, fftfreq

    time_ns = np.arange(data.shape[1]) * dt * 1e9
    angles = START_ANGLE_DEG + np.arange(N_ANGLES) * ANGLE_STEP_DEG

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'UWCEM {PHANTOM_ID} Multi-Angle Scan (72 angles, 2D TMz)\n'
                 f'Tumor: {TUMOR_DIAMETER_MM}mm at '
                 f'({tumor_info["tum_x"]:.1f}, {tumor_info["tum_y"]:.1f}) cm',
                 fontsize=14)

    # 1. Time-domain sinogram
    vmax = np.percentile(np.abs(data), 99)
    im = axes[0, 0].imshow(data, aspect='auto', cmap='seismic',
                            extent=[time_ns[0], time_ns[-1], angles[-1], angles[0]],
                            vmin=-vmax, vmax=vmax)
    axes[0, 0].set_xlabel('Time (ns)')
    axes[0, 0].set_ylabel('Angle (deg)')
    axes[0, 0].set_title('Time-domain sinogram')
    plt.colorbar(im, ax=axes[0, 0], label='Ez (V/m)')

    # 2. Frequency-domain sinogram
    N = data.shape[1]
    freqs = fftfreq(N, dt)
    pos = freqs > 0
    data_fft = np.abs(fft(data, axis=1)[:, pos])
    freqs_ghz = freqs[pos] / 1e9
    band = freqs_ghz <= 10

    im2 = axes[0, 1].imshow(20*np.log10(data_fft[:, band] + 1e-30),
                              aspect='auto', cmap='inferno',
                              extent=[freqs_ghz[band][0], freqs_ghz[band][-1],
                                      angles[-1], angles[0]])
    axes[0, 1].set_xlabel('Frequency (GHz)')
    axes[0, 1].set_ylabel('Angle (deg)')
    axes[0, 1].set_title('Frequency-domain sinogram (dB)')
    plt.colorbar(im2, ax=axes[0, 1])

    # 3. Example traces at different angles
    for idx in [0, 18, 36, 54]:
        axes[0, 2].plot(time_ns, data[idx], linewidth=0.8,
                        label=f'{angles[idx]:.0f} deg')
    axes[0, 2].set_xlabel('Time (ns)')
    axes[0, 2].set_ylabel('Ez (V/m)')
    axes[0, 2].set_title('Selected traces')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Angular variation of peak amplitude
    peak_amps = np.max(np.abs(data), axis=1)
    axes[1, 0].plot(angles, peak_amps, 'b.-')
    axes[1, 0].set_xlabel('Angle (deg)')
    axes[1, 0].set_ylabel('Peak |Ez| (V/m)')
    axes[1, 0].set_title('Peak amplitude vs angle')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Energy per angle
    energy = np.sum(data**2, axis=1) * dt
    axes[1, 1].plot(angles, energy, 'r.-')
    axes[1, 1].set_xlabel('Angle (deg)')
    axes[1, 1].set_ylabel('Energy (a.u.)')
    axes[1, 1].set_title('Received energy vs angle')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Frequency content at 1-8 GHz band
    band_mask = (freqs_ghz >= 1) & (freqs_ghz <= 8)
    band_energy = np.sum(data_fft[:, band_mask]**2, axis=1)
    axes[1, 2].plot(angles, band_energy / band_energy.max(), 'g.-')
    axes[1, 2].set_xlabel('Angle (deg)')
    axes[1, 2].set_ylabel('Normalized energy')
    axes[1, 2].set_title('In-band energy (1-8 GHz)')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "multiscan_sinogram.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"UWCEM Multi-Angle Scan: {PHANTOM_ID}")
    print(f"  Slice: z={SLICE_IDX}, Resolution: {DX*1000:.0f}mm")
    print(f"  Angles: {N_ANGLES}, Antenna radius: {ANTENNA_RADIUS_M*100:.1f}cm")
    print("=" * 60)

    # Step 1: Prepare geometry
    print("\n[1/4] Preparing geometry...")
    domain_size, center, tumor_info = prepare_geometry()
    print(f"  Domain: {domain_size[0]*100:.1f} x {domain_size[1]*100:.1f} cm")
    if tumor_info:
        print(f"  Tumor: {tumor_info['diameter_mm']}mm at "
              f"({tumor_info['tum_x']:.1f}, {tumor_info['tum_y']:.1f}) cm")

    # Step 2: Generate input files
    print("\n[2/4] Generating input files...")
    input_files = generate_input_files(domain_size, center)
    print(f"  Generated {len(input_files)} .in files")

    # Step 3: Run simulations
    print(f"\n[3/4] Running {N_ANGLES} simulations...")
    run_all(input_files)

    # Step 4: Collect and visualize
    print("\n[4/4] Collecting results...")
    data, dt = collect_results()

    np.savez(os.path.join(WORK_DIR, "multiscan_result.npz"),
             data=data, dt=dt, n_angles=N_ANGLES,
             phantom_id=PHANTOM_ID, slice_idx=SLICE_IDX,
             tumor_info=str(tumor_info))
    print(f"  Saved: multiscan_result.npz")

    print("\nGenerating visualizations...")
    visualize(data, dt, tumor_info)
    print("\nDone!")


if __name__ == "__main__":
    main()
