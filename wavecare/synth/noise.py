"""
Noise models for post-simulation processing.

Applies realistic noise to simulated S-parameter data to bridge
the sim-to-real gap.
"""

import numpy as np


def add_awgn(s_params, snr_db=30.0, rng=None):
    """Add complex additive white Gaussian noise.

    Parameters
    ----------
    s_params : ndarray (complex)
        S-parameter data, shape (n_measurements, n_freqs) or (n_scans, n_measurements, n_freqs).
    snr_db : float
        Signal-to-noise ratio in dB.
    rng : numpy.random.Generator or None

    Returns
    -------
    noisy : ndarray (complex)
        S-parameters with added noise.
    """
    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(np.abs(s_params) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power / 2)  # /2 for real+imag parts

    noise = rng.normal(0, noise_std, s_params.shape) + \
            1j * rng.normal(0, noise_std, s_params.shape)

    return s_params + noise


def add_phase_noise(s_params, phase_std_deg=2.0, rng=None):
    """Add random phase perturbation.

    Simulates VNA phase measurement errors and cable instabilities.

    Parameters
    ----------
    s_params : ndarray (complex)
        S-parameter data.
    phase_std_deg : float
        Standard deviation of phase noise in degrees.
    rng : numpy.random.Generator or None

    Returns
    -------
    noisy : ndarray (complex)
    """
    if rng is None:
        rng = np.random.default_rng()

    phase_noise = rng.normal(0, np.deg2rad(phase_std_deg), s_params.shape)
    return s_params * np.exp(1j * phase_noise)


def add_amplitude_noise(s_params, amp_std_db=0.2, rng=None):
    """Add multiplicative amplitude noise.

    Simulates VNA amplitude measurement uncertainty.

    Parameters
    ----------
    s_params : ndarray (complex)
        S-parameter data.
    amp_std_db : float
        Standard deviation of amplitude noise in dB.
    rng : numpy.random.Generator or None

    Returns
    -------
    noisy : ndarray (complex)
    """
    if rng is None:
        rng = np.random.default_rng()

    amp_noise_db = rng.normal(0, amp_std_db, s_params.shape)
    amp_factor = 10 ** (amp_noise_db / 20)
    return s_params * amp_factor


def apply_noise_model(s_params, snr_db=30.0, phase_std_deg=2.0,
                      amp_std_db=0.2, rng=None):
    """Apply the full noise model (AWGN + phase + amplitude).

    Parameters
    ----------
    s_params : ndarray (complex)
        Clean simulated S-parameters.
    snr_db : float
        SNR for additive noise (20-50 dB typical).
    phase_std_deg : float
        Phase noise std (1-3 deg typical).
    amp_std_db : float
        Amplitude noise std (0.1-0.4 dB typical).
    rng : numpy.random.Generator or None

    Returns
    -------
    noisy : ndarray (complex)
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy = add_awgn(s_params, snr_db=snr_db, rng=rng)
    noisy = add_phase_noise(noisy, phase_std_deg=phase_std_deg, rng=rng)
    noisy = add_amplitude_noise(noisy, amp_std_db=amp_std_db, rng=rng)

    return noisy


def random_noise_params(rng=None):
    """Sample random noise parameters for domain randomization.

    Returns
    -------
    params : dict
        'snr_db', 'phase_std_deg', 'amp_std_db'
    """
    if rng is None:
        rng = np.random.default_rng()

    return {
        "snr_db": rng.uniform(20.0, 50.0),
        "phase_std_deg": rng.uniform(0.5, 3.0),
        "amp_std_db": rng.uniform(0.05, 0.4),
    }
