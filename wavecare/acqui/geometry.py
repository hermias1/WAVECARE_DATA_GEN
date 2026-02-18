"""Parametric antenna array geometry for microwave imaging.

Supports circular arrays with configurable number of antennas,
radius, monostatic/bistatic/multistatic modes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ArrayGeometry:
    """Circular antenna array configuration.

    Parameters
    ----------
    n_antennas : int
        Number of antenna positions in the array.
    radius_m : float
        Array radius in meters.
    start_angle_deg : float
        Starting angle of first antenna (degrees, CCW from +x).
    mode : str
        "monostatic" (S11), "bistatic" (fixed Tx-Rx offset),
        or "multistatic" (all pairs).
    bistatic_offset_deg : float
        Tx-Rx angular offset for bistatic mode.
    freq_start_hz : float
        Start frequency.
    freq_stop_hz : float
        Stop frequency.
    n_freqs : int
        Number of frequency points.
    center_m : tuple
        (x, y) center of the array in meters.
    """
    n_antennas: int = 72
    radius_m: float = 0.075
    start_angle_deg: float = -130.0
    mode: str = "bistatic"
    bistatic_offset_deg: float = 60.0
    freq_start_hz: float = 1e9
    freq_stop_hz: float = 8e9
    n_freqs: int = 1001
    center_m: Tuple[float, float] = (0.0, 0.0)

    @property
    def angle_step_deg(self):
        return 360.0 / self.n_antennas

    @property
    def angles_deg(self):
        return self.start_angle_deg + np.arange(self.n_antennas) * self.angle_step_deg

    @property
    def angles_rad(self):
        return np.deg2rad(self.angles_deg)

    @property
    def freqs_hz(self):
        return np.linspace(self.freq_start_hz, self.freq_stop_hz, self.n_freqs)

    def antenna_positions(self) -> np.ndarray:
        """Compute (x, y) positions of all antennas in meters.

        Returns shape (n_antennas, 2).
        """
        cx, cy = self.center_m
        angles = self.angles_rad
        x = cx + self.radius_m * np.cos(angles)
        y = cy + self.radius_m * np.sin(angles)
        return np.column_stack([x, y])

    def tx_rx_pairs(self) -> List[Tuple[int, int]]:
        """Return list of (tx_idx, rx_idx) pairs for the scan.

        For monostatic: [(0,0), (1,1), ...]
        For bistatic: [(0, offset), (1, 1+offset), ...]
        For multistatic: all N*(N-1)/2 pairs.
        """
        n = self.n_antennas

        if self.mode == "monostatic":
            return [(i, i) for i in range(n)]

        elif self.mode == "bistatic":
            offset = int(round(self.bistatic_offset_deg / self.angle_step_deg))
            return [(i, (i + offset) % n) for i in range(n)]

        elif self.mode == "multistatic":
            pairs = []
            for tx in range(n):
                for rx in range(tx + 1, n):
                    pairs.append((tx, rx))
            return pairs

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def n_measurements(self) -> int:
        return len(self.tx_rx_pairs())
