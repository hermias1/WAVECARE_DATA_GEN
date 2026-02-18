"""
Dielectric property models for breast tissues.

Implements Cole-Cole and Debye models based on Lazebnik et al. (2007)
and Gabriel et al. (1996) for breast, skin, and muscle tissues.

References:
- Lazebnik et al., Phys. Med. Biol. 52:2637 (2007) [normal tissue]
- Lazebnik et al., Phys. Med. Biol. 52:6093 (2007) [malignant tissue]
- Gabriel et al., Phys. Med. Biol. 41:2271 (1996) [other tissues]
"""

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
EPS_0 = 8.854187817e-12  # vacuum permittivity (F/m)

# --------------------------------------------------------------------------- #
# Cole-Cole 1-pole model: Lazebnik et al. 2007, Table I (0.5 - 20 GHz)
# Keys: eps_inf, delta_eps, tau_ps, alpha, sigma_s
# --------------------------------------------------------------------------- #
LAZEBNIK_COLE_COLE = {
    # Adipose tissue (Group 3: 85-100% adipose)
    "adipose_low": dict(eps_inf=2.908, delta_eps=1.200, tau_ps=16.88, alpha=0.069, sigma_s=0.020),
    "adipose_med": dict(eps_inf=3.140, delta_eps=1.708, tau_ps=14.65, alpha=0.061, sigma_s=0.036),
    "adipose_high": dict(eps_inf=4.031, delta_eps=3.654, tau_ps=14.12, alpha=0.055, sigma_s=0.083),
    # Transitional (Group 2: 31-84% adipose)
    "transitional_low": dict(eps_inf=4.717, delta_eps=7.280, tau_ps=13.14, alpha=0.047, sigma_s=0.130),
    "transitional_med": dict(eps_inf=5.573, delta_eps=16.55, tau_ps=12.19, alpha=0.035, sigma_s=0.304),
    "transitional_high": dict(eps_inf=6.883, delta_eps=30.83, tau_ps=11.22, alpha=0.027, sigma_s=0.549),
    # Fibroglandular (Group 1: 0-30% adipose)
    "fibroglandular_low": dict(eps_inf=9.941, delta_eps=26.60, tau_ps=10.90, alpha=0.003, sigma_s=0.462),
    "fibroglandular_med": dict(eps_inf=7.821, delta_eps=41.48, tau_ps=10.66, alpha=0.047, sigma_s=0.713),
    "fibroglandular_high": dict(eps_inf=6.151, delta_eps=48.26, tau_ps=10.26, alpha=0.049, sigma_s=0.809),
    # Malignant tumor (from Lazebnik 2007b, approximate median)
    "tumor_malignant": dict(eps_inf=7.0, delta_eps=50.0, tau_ps=10.0, alpha=0.05, sigma_s=0.90),
    # Benign tumor (within fibroglandular range)
    "tumor_benign": dict(eps_inf=8.5, delta_eps=35.0, tau_ps=10.8, alpha=0.04, sigma_s=0.60),
}

# Gabriel et al. 1996 - 1-pole approximation for 1-10 GHz
GABRIEL_COLE_COLE = {
    "skin_dry": dict(eps_inf=4.0, delta_eps=32.0, tau_ps=7.234, alpha=0.0, sigma_s=0.0002),
    "muscle": dict(eps_inf=4.0, delta_eps=50.0, tau_ps=7.234, alpha=0.1, sigma_s=0.2),
}

# Debye 1-pole (3-10 GHz) - fixed tau = 13.0 ps (Lazebnik 2007c)
LAZEBNIK_DEBYE = {
    "adipose_med": dict(eps_inf=2.55, delta_eps=1.85, tau_ps=13.0, sigma_s=0.043),
    "fibroglandular_med": dict(eps_inf=13.81, delta_eps=35.55, tau_ps=13.0, sigma_s=0.738),
    "tumor_malignant": dict(eps_inf=14.0, delta_eps=42.0, tau_ps=13.0, sigma_s=0.90),
}

# All tissue types recognized by the scene generator
ALL_TISSUES = list(LAZEBNIK_COLE_COLE.keys()) + list(GABRIEL_COLE_COLE.keys())

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def cole_cole_permittivity(freqs_hz, eps_inf, delta_eps, tau_ps, alpha,
                           sigma_s):
    """Compute complex permittivity using 1-pole Cole-Cole model.

    Parameters
    ----------
    freqs_hz : array_like
        Frequencies in Hz.
    eps_inf : float
        High-frequency permittivity limit.
    delta_eps : float
        Dielectric decrement.
    tau_ps : float
        Relaxation time in picoseconds.
    alpha : float
        Cole-Cole broadening parameter (0 = Debye).
    sigma_s : float
        Static ionic conductivity (S/m).

    Returns
    -------
    eps_complex : ndarray (complex128)
        Complex relative permittivity at each frequency.
    """
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    omega = 2 * np.pi * freqs_hz
    tau = tau_ps * 1e-12  # convert ps -> s

    eps_complex = (
        eps_inf
        + delta_eps / (1.0 + (1j * omega * tau) ** (1.0 - alpha))
        + sigma_s / (1j * omega * EPS_0)
    )
    return eps_complex


def debye_permittivity(freqs_hz, eps_inf, delta_eps, tau_ps, sigma_s):
    """Compute complex permittivity using 1-pole Debye model.

    Equivalent to Cole-Cole with alpha=0.
    """
    return cole_cole_permittivity(freqs_hz, eps_inf, delta_eps, tau_ps,
                                  alpha=0.0, sigma_s=sigma_s)


def get_tissue_permittivity(tissue_name, freqs_hz, model="cole_cole",
                            perturbation=0.0, rng=None):
    """Get complex permittivity for a named tissue.

    Parameters
    ----------
    tissue_name : str
        One of the keys in LAZEBNIK_COLE_COLE or GABRIEL_COLE_COLE.
    freqs_hz : array_like
        Frequencies in Hz.
    model : str
        "cole_cole" or "debye".
    perturbation : float
        Relative std of Gaussian perturbation on each parameter
        (0.0 = no perturbation, 0.1 = 10%).
    rng : numpy.random.Generator or None
        Random generator for perturbation reproducibility.

    Returns
    -------
    eps_complex : ndarray (complex128)
    """
    if model == "debye" and tissue_name in LAZEBNIK_DEBYE:
        params = dict(LAZEBNIK_DEBYE[tissue_name])
        params["alpha"] = 0.0
    elif tissue_name in LAZEBNIK_COLE_COLE:
        params = dict(LAZEBNIK_COLE_COLE[tissue_name])
    elif tissue_name in GABRIEL_COLE_COLE:
        params = dict(GABRIEL_COLE_COLE[tissue_name])
    else:
        raise ValueError(f"Unknown tissue: {tissue_name}. "
                         f"Available: {ALL_TISSUES}")

    # Apply Gaussian perturbation to parameters
    if perturbation > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        for key in ["eps_inf", "delta_eps", "tau_ps", "sigma_s"]:
            if key in params:
                factor = 1.0 + rng.normal(0, perturbation)
                params[key] = max(params[key] * factor, 0.0)

    alpha = params.pop("alpha", 0.0)
    return cole_cole_permittivity(freqs_hz, alpha=alpha, **params)


def umbmid_frequencies(n_freqs=1001, ini_f=1e9, fin_f=8e9):
    """Return the UMBMID frequency vector."""
    return np.linspace(ini_f, fin_f, n_freqs)


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #

def permittivity_to_real_imag(eps_complex):
    """Split complex permittivity into real part (eps_r) and conductivity."""
    eps_r = np.real(eps_complex)
    # sigma = -omega * eps_0 * imag(eps_complex)  -- but we return raw
    eps_i = -np.imag(eps_complex)
    return eps_r, eps_i


def tissue_summary(tissue_name, freq_ghz=6.0, model="cole_cole"):
    """Print a quick summary of a tissue's properties at a given freq."""
    f = freq_ghz * 1e9
    eps = get_tissue_permittivity(tissue_name, [f], model=model)[0]
    eps_r = np.real(eps)
    sigma_eff = -2 * np.pi * f * EPS_0 * np.imag(eps)
    print(f"{tissue_name} @ {freq_ghz} GHz: "
          f"eps_r = {eps_r:.2f}, sigma_eff = {sigma_eff:.3f} S/m")
    return eps_r, sigma_eff
