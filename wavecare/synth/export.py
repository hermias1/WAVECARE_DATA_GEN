"""
Export synthetic data to UMBMID-compatible format.

Converts gprMax simulation outputs (time-domain) to frequency-domain
S-parameters and packages them in the same format as UMBMID clean files.

NOTE: Uses pickle for UMBMID compatibility (their standard format).
"""

import os
import numpy as np

# pickle is required for UMBMID format compatibility - the upstream
# dataset (UM-BMID) uses pickle as its standard serialization format.
import pickle  # nosec - needed for UMBMID format compatibility


def td_to_fd(td_signals, dt, n_freqs=1001, ini_freq=1e9, fin_freq=8e9):
    """Convert time-domain signals to frequency-domain S-parameters.

    Parameters
    ----------
    td_signals : ndarray, shape (n_time_steps, n_angles) or (n_time_steps,)
        Time-domain voltage/field signals from gprMax.
    dt : float
        Time step in seconds.
    n_freqs : int
        Number of frequency points in output.
    ini_freq : float
        Start frequency in Hz.
    fin_freq : float
        Stop frequency in Hz.

    Returns
    -------
    fd_data : ndarray (complex), shape (n_freqs, n_angles) or (n_freqs,)
        Frequency-domain S-parameters at the target frequencies.
    freqs : ndarray
        Frequency vector in Hz.
    """
    n_steps = td_signals.shape[0]
    freqs_target = np.linspace(ini_freq, fin_freq, n_freqs)

    # FFT of the time-domain signal
    fd_full = np.fft.fft(td_signals, axis=0)
    freq_full = np.fft.fftfreq(n_steps, d=dt)

    # Only keep positive frequencies
    pos_mask = freq_full >= 0
    freq_pos = freq_full[pos_mask]
    fd_pos = fd_full[pos_mask] if td_signals.ndim == 1 else fd_full[pos_mask, :]

    # Interpolate to target frequency grid
    if td_signals.ndim == 1:
        fd_data = np.interp(freqs_target, freq_pos, np.real(fd_pos)) + \
                  1j * np.interp(freqs_target, freq_pos, np.imag(fd_pos))
    else:
        n_angles = td_signals.shape[1]
        fd_data = np.zeros((n_freqs, n_angles), dtype=complex)
        for i in range(n_angles):
            fd_data[:, i] = (
                np.interp(freqs_target, freq_pos, np.real(fd_pos[:, i]))
                + 1j * np.interp(freqs_target, freq_pos, np.imag(fd_pos[:, i]))
            )

    return fd_data, freqs_target


def calibrate(fd_data, fd_reference):
    """Apply calibration by subtracting reference (empty chamber).

    Parameters
    ----------
    fd_data : ndarray (complex), shape (n_freqs, n_angles)
    fd_reference : ndarray (complex), shape (n_freqs, n_angles)

    Returns
    -------
    calibrated : ndarray (complex)
    """
    return fd_data - fd_reference


def package_scan(fd_s11, metadata, fd_s21=None):
    """Package a scan into UMBMID-compatible dict.

    Parameters
    ----------
    fd_s11 : ndarray (complex), shape (1001, 72)
    metadata : dict
    fd_s21 : ndarray (complex) or None

    Returns
    -------
    scan : dict
    """
    scan = {
        "fd_s11": fd_s11,
        "metadata": metadata,
    }
    if fd_s21 is not None:
        scan["fd_s21"] = fd_s21
    return scan


def save_dataset(scans, output_dir, dataset_name="synth"):
    """Save a list of scans as UMBMID-compatible pickle files.

    Creates:
    - fd_data_s11_{dataset_name}.pickle : shape (n_scans, n_freqs, n_angles)
    - md_list_s11_{dataset_name}.pickle : list of metadata dicts

    Uses pickle for UMBMID format compatibility (upstream standard).

    Parameters
    ----------
    scans : list of dict
        Each from package_scan().
    output_dir : str
    dataset_name : str
    """
    os.makedirs(output_dir, exist_ok=True)

    n_scans = len(scans)
    n_freqs = scans[0]["fd_s11"].shape[0]
    n_angles = scans[0]["fd_s11"].shape[1]

    fd_data = np.zeros((n_scans, n_freqs, n_angles), dtype=complex)
    md_list = []

    for i, scan in enumerate(scans):
        fd_data[i] = scan["fd_s11"]
        md_list.append(scan["metadata"])

    # Save S11 data (pickle for UMBMID compat)
    s11_path = os.path.join(output_dir, f"fd_data_s11_{dataset_name}.pickle")
    with open(s11_path, 'wb') as f:
        pickle.dump(fd_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata
    md_path = os.path.join(output_dir, f"md_list_s11_{dataset_name}.pickle")
    with open(md_path, 'wb') as f:
        pickle.dump(md_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    # If S21 data exists
    if scans[0].get("fd_s21") is not None:
        fd_s21 = np.zeros_like(fd_data)
        for i, scan in enumerate(scans):
            fd_s21[i] = scan["fd_s21"]

        s21_path = os.path.join(output_dir, f"fd_data_s21_{dataset_name}.pickle")
        with open(s21_path, 'wb') as f:
            pickle.dump(fd_s21, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {n_scans} scans to {output_dir}")
    print(f"  S11: {s11_path}")
    print(f"  Metadata: {md_path}")

    return {"s11_path": s11_path, "md_path": md_path}
