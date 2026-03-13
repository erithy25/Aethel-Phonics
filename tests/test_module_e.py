"""Tests for Module E: Visualization and Performance Metrics."""

import numpy as np
import pytest

from apld_mps.module_e import FieldVisualiser, PerformanceDashboard, GateMetrics
from apld_mps.module_c.detector import DetectorReading


class TestFieldVisualiser:
    def test_capture_frames(self):
        vis = FieldVisualiser()
        for t in range(5):
            vis.capture_frame(np.random.rand(10, 10), time_ps=t * 0.1)
        assert vis.n_frames == 5

    def test_intensity_range(self):
        vis = FieldVisualiser()
        vis.capture_frame(np.ones((5, 5)) * 2.0, time_ps=0)
        vis.capture_frame(np.ones((5, 5)) * 8.0, time_ps=1)
        vmin, vmax = vis.get_intensity_range()
        assert vmin == pytest.approx(2.0)
        assert vmax == pytest.approx(8.0)


class TestPerformanceDashboard:
    def _make_reading(self, peak: float, arrival_idx: int = 3) -> DetectorReading:
        times = np.arange(10, dtype=np.float64)
        intensity = np.zeros(10)
        intensity[arrival_idx:] = peak
        phase = np.zeros(10)
        return DetectorReading(times_ps=times, intensity=intensity, phase=phase)

    def test_latency(self):
        dash = PerformanceDashboard()
        reading = self._make_reading(peak=1.0, arrival_idx=5)
        lat = dash.compute_latency(reading, input_time_ps=0.0)
        assert lat == pytest.approx(5.0)

    def test_switching_energy(self):
        dash = PerformanceDashboard()
        energy = dash.compute_switching_energy(
            pulse_energies_eV=[1.72],
            pulse_durations_fs=[100.0],
            peak_intensities_W_cm2=[1e4],
            spot_areas_um2=[0.25],
        )
        assert energy > 0

    def test_snr(self):
        dash = PerformanceDashboard()
        snr = dash.compute_snr(intensity_logic_1=100.0, intensity_logic_0=1.0)
        assert snr == pytest.approx(20.0)

    def test_evaluate_gate(self):
        dash = PerformanceDashboard()
        r1 = self._make_reading(peak=10.0, arrival_idx=4)
        r0 = self._make_reading(peak=0.1, arrival_idx=4)
        metrics = dash.evaluate_gate(
            output_reading_1=r1,
            output_reading_0=r0,
            pulse_energies_eV=[1.72],
            pulse_durations_fs=[100.0],
            peak_intensities=[1e4],
            spot_areas_um2=[0.25],
            input_time_ps=0.0,
        )
        assert metrics.switching_latency_ps > 0
        assert metrics.switching_energy_aJ > 0
        assert metrics.snr_dB > 0
