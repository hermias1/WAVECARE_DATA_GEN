"""Compare synthetic data against real UMBMID measurements.

Provides metrics for evaluating how realistic the synthetic data is.
"""

import numpy as np


def signal_statistics(data):
    """Compute basic statistics for a scan dataset.

    Parameters
    ----------
    data : ndarray, shape (n_measurements, n_samples)

    Returns
    -------
    dict with peak, rms, dynamic_range, snr_estimate
    """
    peak = np.max(np.abs(data), axis=1)
    rms = np.sqrt(np.mean(np.abs(data)**2, axis=1))

    return {
        "peak_mean": float(np.mean(peak)),
        "peak_std": float(np.std(peak)),
        "rms_mean": float(np.mean(rms)),
        "dynamic_range_db": float(20 * np.log10(peak.max() / (peak.min() + 1e-30))),
    }


def spectral_similarity(fd_synth, fd_real, freqs_hz):
    """Compare spectral content between synthetic and real data.

    Parameters
    ----------
    fd_synth : ndarray, shape (n_meas, n_freqs), complex
    fd_real : ndarray, shape (n_meas, n_freqs), complex
    freqs_hz : ndarray

    Returns
    -------
    dict with spectral correlation, magnitude difference, phase difference
    """
    mag_synth = np.abs(fd_synth)
    mag_real = np.abs(fd_real)

    # Normalize per measurement
    mag_synth_norm = mag_synth / (np.max(mag_synth, axis=1, keepdims=True) + 1e-30)
    mag_real_norm = mag_real / (np.max(mag_real, axis=1, keepdims=True) + 1e-30)

    # Average spectral shape correlation
    correlations = []
    for i in range(min(mag_synth.shape[0], mag_real.shape[0])):
        a = mag_synth_norm[i]
        b = mag_real_norm[i]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        c = np.corrcoef(a, b)[0, 1]
        if np.isfinite(c):
            correlations.append(c)

    if correlations:
        corr_mean = float(np.mean(correlations))
        corr_std = float(np.std(correlations))
    else:
        corr_mean = float("nan")
        corr_std = float("nan")

    return {
        "spectral_correlation_mean": corr_mean,
        "spectral_correlation_std": corr_std,
        "mag_ratio_db": float(20 * np.log10(
            np.mean(mag_synth) / (np.mean(mag_real) + 1e-30)
        )),
    }


def angular_variation_similarity(data_synth, data_real):
    """Compare angular energy variation patterns.

    Parameters
    ----------
    data_synth, data_real : ndarray, shape (n_angles, n_samples)

    Returns
    -------
    dict with angular correlation
    """
    energy_synth = np.sum(np.abs(data_synth)**2, axis=1)
    energy_real = np.sum(np.abs(data_real)**2, axis=1)

    # Normalize
    energy_synth = energy_synth / energy_synth.max()
    energy_real = energy_real / energy_real.max()

    n = min(len(energy_synth), len(energy_real))
    a = energy_synth[:n]
    b = energy_real[:n]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        corr = float("nan")
    else:
        corr = np.corrcoef(a, b)[0, 1]

    return {
        "angular_energy_correlation": float(corr),
    }
