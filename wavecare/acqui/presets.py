"""Predefined acquisition geometries for known MWI systems."""

from .geometry import ArrayGeometry


def umbmid_gen2() -> ArrayGeometry:
    """UMBMID Generation 2 geometry.

    72 positions, bistatic with +60 deg offset, 1-8 GHz, 1001 points.
    Antenna radius ~7.5 cm. Double-ridged horn antennas.

    Reference: UM-BMID docs/scan_geometry.md
    """
    return ArrayGeometry(
        n_antennas=72,
        radius_m=0.075,
        start_angle_deg=-130.0,
        mode="bistatic",
        bistatic_offset_deg=60.0,
        freq_start_hz=1e9,
        freq_stop_hz=8e9,
        n_freqs=1001,
    )


def maria_m5() -> ArrayGeometry:
    """MARIA M5 clinical system approximation.

    Micrima MARIA: 60 antenna elements, multistatic,
    radar-based imaging, 3-8 GHz.

    Reference: Preece et al., J. Med. Imaging 3(3), 2016
    """
    return ArrayGeometry(
        n_antennas=60,
        radius_m=0.08,
        start_angle_deg=0.0,
        mode="multistatic",
        bistatic_offset_deg=0.0,
        freq_start_hz=3e9,
        freq_stop_hz=8e9,
        n_freqs=501,
    )


def mammowave() -> ArrayGeometry:
    """Mammowave clinical system approximation.

    UBT/Umbria Bioengineering Technologies: 2 antennas rotating,
    bistatic, 1-9 GHz.

    Reference: Tiberi et al., Diagnostics 10(10), 2020
    """
    return ArrayGeometry(
        n_antennas=72,
        radius_m=0.10,
        start_angle_deg=0.0,
        mode="bistatic",
        bistatic_offset_deg=180.0,
        freq_start_hz=1e9,
        freq_stop_hz=9e9,
        n_freqs=1001,
    )


def custom(n_antennas=72, radius_cm=7.5, mode="bistatic",
           offset_deg=60.0, freq_range_ghz=(1, 8), n_freqs=1001) -> ArrayGeometry:
    """Create a custom acquisition geometry."""
    return ArrayGeometry(
        n_antennas=n_antennas,
        radius_m=radius_cm / 100,
        start_angle_deg=0.0,
        mode=mode,
        bistatic_offset_deg=offset_deg,
        freq_start_hz=freq_range_ghz[0] * 1e9,
        freq_stop_hz=freq_range_ghz[1] * 1e9,
        n_freqs=n_freqs,
    )
