"""
End-to-end pipeline tests using synthetic mock data (no gprMax required).

Validates: td_to_fd shape conventions, calibration, noise model,
export/load round-trip, and metadata generation.
"""

import os
import tempfile
import numpy as np
import pytest

from wavecare.synth.export import td_to_fd, calibrate, package_scan, save_dataset
from wavecare.synth.noise import (
    add_awgn, add_phase_noise, add_amplitude_noise,
    apply_noise_model, random_noise_params,
)
from wavecare.synth.scenes import scene_to_metadata
from wavecare.eval.compare import (
    signal_statistics, spectral_similarity, angular_variation_similarity,
)
from wavecare.acqui.presets import umbmid_gen2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_MEAS = 72
N_TIMESTEPS = 3393
DT = 2.36e-12  # typical gprMax dt


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def mock_td(rng):
    """Fake time-domain data: damped sinusoid + noise, (n_meas, n_timesteps)."""
    t = np.arange(N_TIMESTEPS) * DT
    freq = 4.5e9  # center frequency
    envelope = np.exp(-t / 2e-9)
    signal = envelope * np.sin(2 * np.pi * freq * t)
    # Each measurement gets slightly different amplitude
    amps = 1.0 + 0.2 * rng.standard_normal(N_MEAS)
    td = np.outer(amps, signal)
    return td


@pytest.fixture
def mock_td_ref(rng):
    """Reference scan (slightly different from tumor scan)."""
    t = np.arange(N_TIMESTEPS) * DT
    freq = 4.5e9
    envelope = np.exp(-t / 2e-9)
    signal = envelope * np.sin(2 * np.pi * freq * t)
    amps = 1.0 + 0.1 * rng.standard_normal(N_MEAS)
    return np.outer(amps, signal)


# ---------------------------------------------------------------------------
# td_to_fd
# ---------------------------------------------------------------------------

class TestTdToFd:
    def test_2d_shape(self, mock_td):
        fd, freqs = td_to_fd(mock_td, DT)
        assert fd.shape == (N_MEAS, 1001)
        assert freqs.shape == (1001,)
        assert np.iscomplexobj(fd)

    def test_1d_shape(self, mock_td):
        fd, freqs = td_to_fd(mock_td[0], DT)
        assert fd.shape == (1001,)
        assert np.iscomplexobj(fd)

    def test_custom_freq_range(self, mock_td):
        fd, freqs = td_to_fd(mock_td, DT, n_freqs=501,
                              ini_freq=3e9, fin_freq=8e9)
        assert fd.shape == (N_MEAS, 501)
        assert freqs[0] == pytest.approx(3e9)
        assert freqs[-1] == pytest.approx(8e9)

    def test_nonzero_in_band(self, mock_td):
        """Signal at 4.5 GHz should produce non-zero FD content."""
        fd, freqs = td_to_fd(mock_td, DT)
        # Find bin closest to 4.5 GHz
        idx = np.argmin(np.abs(freqs - 4.5e9))
        assert np.mean(np.abs(fd[:, idx])) > 0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_shape_preserved(self, mock_td, mock_td_ref):
        fd_tumor, freqs = td_to_fd(mock_td, DT)
        fd_ref, _ = td_to_fd(mock_td_ref, DT)
        fd_cal = calibrate(fd_tumor, fd_ref)
        assert fd_cal.shape == fd_tumor.shape

    def test_reduces_signal(self, mock_td, mock_td_ref):
        """Calibrated signal should be smaller than raw (common mode removed)."""
        fd_tumor, _ = td_to_fd(mock_td, DT)
        fd_ref, _ = td_to_fd(mock_td_ref, DT)
        fd_cal = calibrate(fd_tumor, fd_ref)
        assert np.mean(np.abs(fd_cal)) < np.mean(np.abs(fd_tumor))

    def test_identical_gives_zero(self, mock_td):
        fd, _ = td_to_fd(mock_td, DT)
        fd_cal = calibrate(fd, fd)
        np.testing.assert_allclose(fd_cal, 0)


# ---------------------------------------------------------------------------
# Noise model
# ---------------------------------------------------------------------------

class TestNoise:
    def test_awgn_adds_noise(self, rng):
        clean = np.ones((10, 100), dtype=complex)
        noisy = add_awgn(clean, snr_db=20.0, rng=rng)
        assert noisy.shape == clean.shape
        assert not np.allclose(noisy, clean)

    def test_phase_noise_preserves_magnitude(self, rng):
        clean = np.ones((10, 100), dtype=complex) * (1 + 0j)
        noisy = add_phase_noise(clean, phase_std_deg=5.0, rng=rng)
        np.testing.assert_allclose(np.abs(noisy), np.abs(clean), atol=1e-10)

    def test_amplitude_noise_preserves_phase(self, rng):
        clean = np.exp(1j * np.linspace(0, 2*np.pi, 100))
        clean = np.tile(clean, (10, 1))
        noisy = add_amplitude_noise(clean, amp_std_db=1.0, rng=rng)
        # Phase should be identical
        np.testing.assert_allclose(np.angle(noisy), np.angle(clean), atol=1e-10)

    def test_full_noise_model(self, rng):
        clean = np.ones((N_MEAS, 100), dtype=complex)
        noisy = apply_noise_model(clean, snr_db=30.0, phase_std_deg=2.0,
                                   amp_std_db=0.2, rng=rng)
        assert noisy.shape == clean.shape
        assert np.iscomplexobj(noisy)

    def test_random_noise_params(self, rng):
        params = random_noise_params(rng=rng)
        assert 20.0 <= params["snr_db"] <= 50.0
        assert 0.5 <= params["phase_std_deg"] <= 3.0
        assert 0.05 <= params["amp_std_db"] <= 0.4


# ---------------------------------------------------------------------------
# Export round-trip (pickle used for UMBMID format compatibility)
# ---------------------------------------------------------------------------

class TestExport:
    def test_package_scan(self):
        fd = np.ones((72, 1001), dtype=complex)
        md = {"id": 0, "source": "test"}
        scan = package_scan(fd, md)
        assert scan["fd_s11"].shape == (72, 1001)
        assert scan["metadata"]["source"] == "test"

    def test_save_load_roundtrip(self, rng):
        # pickle required for UMBMID upstream format compatibility
        import pickle  # nosec B403

        fd1 = rng.standard_normal((72, 1001)) + 1j * rng.standard_normal((72, 1001))
        fd2 = rng.standard_normal((72, 1001)) + 1j * rng.standard_normal((72, 1001))
        scans = [
            package_scan(fd1, {"id": 0, "birads": 1}),
            package_scan(fd2, {"id": 1, "birads": 3}),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dataset(scans, tmpdir, dataset_name="test")

            # Verify files exist
            s11_path = os.path.join(tmpdir, "fd_data_s11_test.pickle")
            md_path = os.path.join(tmpdir, "md_list_s11_test.pickle")
            assert os.path.exists(s11_path)
            assert os.path.exists(md_path)

            # Load and verify shape
            with open(s11_path, 'rb') as f:
                loaded = pickle.load(f)  # nosec B301
            assert loaded.shape == (2, 72, 1001)
            np.testing.assert_allclose(loaded[0], fd1)
            np.testing.assert_allclose(loaded[1], fd2)

            with open(md_path, 'rb') as f:
                md_list = pickle.load(f)  # nosec B301
            assert len(md_list) == 2
            assert md_list[0]["birads"] == 1


# ---------------------------------------------------------------------------
# Eval metrics with complex data
# ---------------------------------------------------------------------------

class TestEvalComplex:
    def test_signal_statistics_complex(self, rng):
        data = rng.standard_normal((10, 100)) + 1j * rng.standard_normal((10, 100))
        stats = signal_statistics(data)
        assert stats["rms_mean"] > 0
        assert not np.isnan(stats["dynamic_range_db"])

    def test_spectral_similarity_identical(self, rng):
        fd = rng.standard_normal((10, 50)) + 1j * rng.standard_normal((10, 50))
        freqs = np.linspace(1e9, 8e9, 50)
        sim = spectral_similarity(fd, fd, freqs)
        assert sim["spectral_correlation_mean"] == pytest.approx(1.0)
        assert sim["mag_ratio_db"] == pytest.approx(0.0, abs=0.1)

    def test_angular_variation_complex(self, rng):
        d1 = rng.standard_normal((10, 100)) + 1j * rng.standard_normal((10, 100))
        d2 = d1 * 2  # same pattern, different scale
        sim = angular_variation_similarity(d1, d2)
        assert sim["angular_energy_correlation"] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Metadata generation
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_tumor_metadata(self):
        scene = {
            "has_tumor": True,
            "tumor_info": {
                "tum_rad": 0.6, "tum_x": 0.1, "tum_y": -0.2,
                "tum_z": 0.0, "tum_diam": 1.2, "shape": "sphere",
            },
            "birads": 2,
            "fib_fraction": 0.35,
        }
        md = scene_to_metadata(scene, scan_id=5, phant_id="071904")
        assert md["tum_rad"] == 0.6
        assert md["phant_id"] == "071904"
        assert md["n_expt"] == 5
        assert md["source"] == "synthetic"

    def test_no_tumor_metadata(self):
        scene = {
            "has_tumor": False,
            "tumor_info": None,
            "birads": 1,
            "fib_fraction": 0.1,
        }
        md = scene_to_metadata(scene, scan_id=0)
        assert np.isnan(md["tum_rad"])
        assert md["tum_shape"] == ""


# ---------------------------------------------------------------------------
# Full chain integration
# ---------------------------------------------------------------------------

class TestFullChain:
    """End-to-end: mock TD -> FD -> calibrate -> noise -> export."""

    def test_full_pipeline(self, mock_td, mock_td_ref, rng):
        # 1. TD -> FD
        fd_tumor, freqs = td_to_fd(mock_td, DT)
        fd_ref, _ = td_to_fd(mock_td_ref, DT)
        assert fd_tumor.shape == (N_MEAS, 1001)

        # 2. Calibrate
        fd_cal = calibrate(fd_tumor, fd_ref)
        assert fd_cal.shape == fd_tumor.shape
        assert np.mean(np.abs(fd_cal)) < np.mean(np.abs(fd_tumor))

        # 3. Noise
        fd_noisy = apply_noise_model(fd_cal, snr_db=30.0, rng=rng)
        assert fd_noisy.shape == fd_cal.shape

        # 4. Package + export
        md = {"id": 0, "birads": 1, "source": "synthetic"}
        scan = package_scan(fd_noisy, md)

        # pickle required for UMBMID upstream format compatibility
        import pickle  # nosec B403

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dataset([scan], tmpdir)

            with open(os.path.join(tmpdir, "fd_data_s11_synth.pickle"), 'rb') as f:
                loaded = pickle.load(f)  # nosec B301

            assert loaded.shape == (1, N_MEAS, 1001)
            np.testing.assert_allclose(loaded[0], fd_noisy)

    def test_umbmid_preset_compatibility(self, mock_td, mock_td_ref, rng):
        """Verify output matches UMBMID expected dimensions."""
        geo = umbmid_gen2()

        # Use preset freq grid
        fd_tumor, freqs = td_to_fd(mock_td, DT, n_freqs=geo.n_freqs,
                                    ini_freq=geo.freq_start_hz,
                                    fin_freq=geo.freq_stop_hz)
        fd_ref, _ = td_to_fd(mock_td_ref, DT, n_freqs=geo.n_freqs,
                              ini_freq=geo.freq_start_hz,
                              fin_freq=geo.freq_stop_hz)

        fd_cal = calibrate(fd_tumor, fd_ref)
        fd_noisy = apply_noise_model(fd_cal, snr_db=30.0, rng=rng)

        # UMBMID: 72 measurements x 1001 freq points
        assert fd_noisy.shape == (72, 1001)
        assert freqs[0] == pytest.approx(1e9)
        assert freqs[-1] == pytest.approx(8e9)
