"""
Quick validation of the dielectric module.

Checks that tissue permittivities are in physically plausible ranges
at key frequencies (1 GHz, 4 GHz, 8 GHz).
"""

import numpy as np

from wavecare.synth.dielectrics import (
    get_tissue_permittivity, umbmid_frequencies, tissue_summary,
    cole_cole_permittivity, LAZEBNIK_COLE_COLE, GABRIEL_COLE_COLE,
)


def test_basic_ranges():
    """Check permittivities are in expected ranges at 6 GHz."""
    f = 6e9

    # Adipose: eps_r should be ~3-5
    eps = get_tissue_permittivity("adipose_med", [f])
    eps_r = np.real(eps[0])
    assert 2 < eps_r < 8, f"adipose eps_r={eps_r} out of range"

    # Fibroglandular: eps_r should be ~30-55
    eps = get_tissue_permittivity("fibroglandular_med", [f])
    eps_r = np.real(eps[0])
    assert 20 < eps_r < 60, f"fibroglandular eps_r={eps_r} out of range"

    # Tumor: eps_r should be ~40-60
    eps = get_tissue_permittivity("tumor_malignant", [f])
    eps_r = np.real(eps[0])
    assert 30 < eps_r < 70, f"tumor eps_r={eps_r} out of range"

    # Skin: eps_r should be ~25-40
    eps = get_tissue_permittivity("skin_dry", [f])
    eps_r = np.real(eps[0])
    assert 15 < eps_r < 50, f"skin eps_r={eps_r} out of range"

    print("PASS: All basic permittivity ranges correct at 6 GHz")


def test_frequency_dispersion():
    """Check that permittivity decreases with frequency (normal dispersion)."""
    freqs = [1e9, 4e9, 8e9]

    for tissue in ["adipose_med", "fibroglandular_med", "tumor_malignant"]:
        eps = get_tissue_permittivity(tissue, freqs)
        eps_r = np.real(eps)

        # Real part should generally decrease with frequency
        assert eps_r[0] >= eps_r[-1], \
            f"{tissue}: eps_r not decreasing ({eps_r[0]:.1f} -> {eps_r[-1]:.1f})"

    print("PASS: Normal dispersion (eps_r decreases with frequency)")


def test_contrast():
    """Check tumor-to-adipose contrast is ~10:1 as expected."""
    f = 6e9

    eps_fat = get_tissue_permittivity("adipose_med", [f])
    eps_tumor = get_tissue_permittivity("tumor_malignant", [f])

    contrast = np.real(eps_tumor[0]) / np.real(eps_fat[0])
    assert 5 < contrast < 20, f"Tumor/adipose contrast={contrast:.1f} unexpected"

    print(f"PASS: Tumor/adipose contrast = {contrast:.1f}:1 at 6 GHz")


def test_perturbation():
    """Check that perturbation changes values but stays in range."""
    f = 6e9
    rng = np.random.default_rng(42)

    eps_base = get_tissue_permittivity("fibroglandular_med", [f])
    eps_perturbed = get_tissue_permittivity("fibroglandular_med", [f],
                                            perturbation=0.1, rng=rng)

    # Should be different
    assert not np.allclose(eps_base, eps_perturbed), \
        "Perturbation did not change values"

    # Should still be in plausible range (within ~50% of base)
    ratio = np.abs(np.real(eps_perturbed[0]) / np.real(eps_base[0]))
    assert 0.5 < ratio < 2.0, f"Perturbation too large: ratio={ratio:.2f}"

    print("PASS: Perturbation works and stays in range")


def test_full_spectrum():
    """Print a full spectrum table for visual inspection."""
    freqs = umbmid_frequencies()
    print(f"\nFull spectrum check ({len(freqs)} points, {freqs[0]/1e9:.1f}-{freqs[-1]/1e9:.1f} GHz)")
    print("-" * 70)

    for tissue in ["adipose_med", "fibroglandular_med", "tumor_malignant",
                   "skin_dry", "muscle"]:
        for f_ghz in [1.0, 4.0, 8.0]:
            tissue_summary(tissue, freq_ghz=f_ghz)
        print()


if __name__ == "__main__":
    test_basic_ranges()
    test_frequency_dispersion()
    test_contrast()
    test_perturbation()
    test_full_spectrum()
    print("\nAll tests passed!")
