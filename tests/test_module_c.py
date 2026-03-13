"""Tests for Module C: Input/Output and Detection."""

import numpy as np
import pytest

from apld_mps.module_c import LaserSource, LaserPulse, PolarisationState, VirtualDetector


class TestLaserPulse:
    def test_wavelength_reasonable(self):
        pulse = LaserPulse(energy_eV=1.72)
        # 1.72 eV ≈ 720 nm
        assert 700 < pulse.wavelength_nm < 740

    def test_temporal_envelope_peak(self):
        pulse = LaserPulse(energy_eV=1.72, duration_fs=100, delay_ps=5.0)
        t = np.array([5.0])
        assert pulse.temporal_envelope(t) == pytest.approx(1.0)

    def test_spatial_profile_peak(self):
        pulse = LaserPulse(energy_eV=1.72, position_nm=(100.0, 200.0))
        x = np.array([100.0])
        y = np.array([200.0])
        assert pulse.spatial_profile(x, y) == pytest.approx(1.0)


class TestLaserSource:
    def test_add_pulse(self):
        src = LaserSource()
        src.add_pulse(LaserPulse(energy_eV=1.72))
        assert src.n_sources == 1

    def test_xor_pair(self):
        src = LaserSource.xor_pair(
            energy_eV=1.72,
            position_a=(0, 1000),
            position_b=(0, -1000),
        )
        assert src.n_sources == 2


class TestVirtualDetector:
    def test_record_and_read(self):
        det = VirtualDetector("out", grid_position=(5, 5), dt_ps=0.1)
        for i in range(10):
            det.record(complex(np.sin(i * 0.5), np.cos(i * 0.5)))
        reading = det.get_reading()
        assert len(reading.times_ps) == 10
        assert reading.peak_intensity > 0

    def test_arrival_time(self):
        det = VirtualDetector("out", grid_position=(5, 5), dt_ps=1.0)
        # Zero for first 3 steps, then signal
        for _ in range(3):
            det.record(0.0 + 0j)
        for _ in range(5):
            det.record(1.0 + 0j)
        reading = det.get_reading()
        assert reading.arrival_time_ps == pytest.approx(3.0)
