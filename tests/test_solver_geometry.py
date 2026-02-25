"""Geometry safety checks for antenna placement."""

import tempfile

import numpy as np
import pytest

from wavecare.acqui.geometry import ArrayGeometry
from wavecare.synth.solver import (
    evaluate_array_placement,
    generate_gprmax_inputs,
    generate_gprmax_inputs_3d,
)


def _disk_mask(nx, ny, cx, cy, radius_cells):
    xx, yy = np.mgrid[:nx, :ny]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_cells ** 2


class TestAntennaPlacementValidation2D:
    def test_raises_when_antennas_are_in_tissue(self):
        dx = 0.001
        labels = np.zeros((80, 80), dtype=np.int8)
        labels[_disk_mask(80, 80, 40, 40, 12)] = 8

        geo_info = {
            "labels_2d": labels,
            "domain_m": (0.08, 0.08),
            "center_m": (0.04, 0.04),
            "geo_shape": labels.shape,
            "dx": dx,
            "pad_cells": 0,
            "slice_idx": 0,
        }
        array_geo = ArrayGeometry(
            n_antennas=8,
            radius_m=0.010,  # inside tissue disk
            start_angle_deg=0.0,
            mode="bistatic",
            bistatic_offset_deg=45.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="antennas are in tissue"):
                generate_gprmax_inputs(
                    geo_info,
                    array_geo,
                    tmpdir,
                    validate_placement=True,
                    center_mode="volume",
                )

    def test_passes_when_antennas_are_in_coupling_with_clearance(self):
        dx = 0.001
        labels = np.zeros((80, 80), dtype=np.int8)
        labels[_disk_mask(80, 80, 40, 40, 12)] = 8

        geo_info = {
            "labels_2d": labels,
            "domain_m": (0.08, 0.08),
            "center_m": (0.04, 0.04),
            "geo_shape": labels.shape,
            "dx": dx,
            "pad_cells": 0,
            "slice_idx": 0,
        }
        array_geo = ArrayGeometry(
            n_antennas=8,
            radius_m=0.025,  # well outside tissue disk
            start_angle_deg=0.0,
            mode="bistatic",
            bistatic_offset_deg=45.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            inputs = generate_gprmax_inputs(
                geo_info,
                array_geo,
                tmpdir,
                validate_placement=True,
                min_clearance_m=0.003,
            )
        assert len(inputs) == array_geo.n_measurements()

    def test_ring_fit_recenters_array_for_offcenter_geometry(self):
        dx = 0.001
        labels = np.zeros((100, 100), dtype=np.int8)
        labels[_disk_mask(100, 100, 60, 50, 20)] = 8

        geo_info = {
            "labels_2d": labels,
            "domain_m": (0.1, 0.1),
            "center_m": (0.05, 0.05),  # geometric center, not tissue center
            "geo_shape": labels.shape,
            "dx": dx,
            "pad_cells": 0,
            "slice_idx": 0,
        }
        array_geo = ArrayGeometry(
            n_antennas=16,
            radius_m=0.025,
            start_angle_deg=0.0,
            mode="bistatic",
            bistatic_offset_deg=45.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="antennas are in tissue"):
                generate_gprmax_inputs(
                    geo_info,
                    array_geo,
                    tmpdir,
                    validate_placement=True,
                    center_mode="volume",
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            inputs = generate_gprmax_inputs(
                geo_info,
                array_geo,
                tmpdir,
                validate_placement=True,
                center_mode="ring_fit",
                center_search_m=0.02,
            )
            assert len(inputs) == array_geo.n_measurements()
            sx, sy = geo_info["array_center_shift_mm"]
            # Expected ~+10 mm shift on x for this synthetic setup.
            assert sx >= 4.5
            assert abs(sy) < 5.0


class TestAntennaPlacementValidation3D:
    def test_raises_for_3d_when_antennas_are_in_tissue(self):
        dx = 0.001
        labels = np.zeros((80, 80, 5), dtype=np.int8)
        mask = _disk_mask(80, 80, 40, 40, 12)
        labels[:, :, 2][mask] = 8

        geo_info = {
            "labels_3d": labels,
            "domain_m": (0.08, 0.08, 0.005),
            "center_m": (0.04, 0.04, 0.002),
            "geo_shape": labels.shape,
            "dx": dx,
            "pad_cells": 0,
        }
        array_geo = ArrayGeometry(
            n_antennas=8,
            radius_m=0.010,
            start_angle_deg=0.0,
            mode="bistatic",
            bistatic_offset_deg=45.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="antennas are in tissue"):
                generate_gprmax_inputs_3d(
                    geo_info,
                    array_geo,
                    tmpdir,
                    validate_placement=True,
                    center_mode="volume",
                )


class TestPlacementAudit:
    def test_evaluate_array_placement_reports_valid_config(self):
        dx = 0.001
        labels = np.zeros((80, 80), dtype=np.int8)
        labels[_disk_mask(80, 80, 40, 40, 12)] = 8

        geo_info = {
            "labels_2d": labels,
            "domain_m": (0.08, 0.08),
            "center_m": (0.04, 0.04),
            "geo_shape": labels.shape,
            "dx": dx,
            "pad_cells": 0,
            "slice_idx": 0,
        }
        array_geo = ArrayGeometry(
            n_antennas=8,
            radius_m=0.025,
            start_angle_deg=0.0,
            mode="bistatic",
            bistatic_offset_deg=45.0,
        )

        rep = evaluate_array_placement(
            geo_info,
            array_geo,
            min_clearance_m=0.003,
            center_mode="volume",
        )
        assert rep["valid"] is True
        assert rep["n_oob"] == 0
        assert rep["n_in_tissue"] == 0
        assert rep["min_clearance_m"] >= 0.003
