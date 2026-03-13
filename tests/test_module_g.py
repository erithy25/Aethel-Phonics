"""Tests for Module G: Quantum-Cosmic Stress Test Suite."""

import numpy as np
import pytest

from apld_mps.module_g import (
    CosmicRaySimulator,
    CosmicRayEvent,
    SelfHealingRouter,
    PhaseStabilityAnalyser,
    VibrationProfile,
    StabilityReport,
)


class TestCosmicRaySimulator:
    def test_generate_events(self):
        sim = CosmicRaySimulator(
            volume_nm=(1_000_000, 1_000_000, 1_000_000),
            flux_per_cm2_s=100.0,  # high flux for testing
        )
        events = sim.generate_events(duration_s=1.0, seed=42)
        assert len(events) > 0
        for ev in events:
            assert ev.energy_MeV > 0
            assert ev.affected_radius_nm > 0
            assert 0 <= ev.coherence_loss <= 1

    def test_error_rate(self):
        sim = CosmicRaySimulator(
            volume_nm=(1_000_000, 1_000_000, 1_000_000),
            flux_per_cm2_s=100.0,
        )
        sim.generate_events(duration_s=1.0, seed=42)
        rate = sim.error_rate(total_operations=1_000_000)
        assert 0 <= rate <= 1

    def test_apply_to_field(self):
        sim = CosmicRaySimulator(volume_nm=(1000, 1000, 1000))
        event = CosmicRayEvent(
            position_nm=(500, 500, 500),
            energy_MeV=100.0,
            affected_radius_nm=200.0,
            coherence_loss=0.5,
        )
        field = np.ones((10, 10, 10), dtype=np.complex128)
        result = sim.apply_to_field(field, grid_spacing_nm=100.0, events=[event])
        # Some cells should be attenuated
        assert np.min(np.abs(result)) < 1.0

    def test_apply_to_2d_field(self):
        sim = CosmicRaySimulator(volume_nm=(1000, 1000, 0))
        event = CosmicRayEvent(
            position_nm=(500, 500, 0),
            energy_MeV=100.0,
            affected_radius_nm=200.0,
            coherence_loss=0.8,
        )
        field = np.ones((10, 10), dtype=np.complex128)
        result = sim.apply_to_field(field, grid_spacing_nm=100.0, events=[event])
        assert np.min(np.abs(result)) < 1.0


class TestSelfHealingRouter:
    def test_find_path_no_disruption(self):
        router = SelfHealingRouter(
            volume_nm=(10000, 10000, 10000),
            grid_resolution_nm=2000,
        )
        path = router.find_path(
            start_nm=(0, 0, 0),
            end_nm=(9000, 9000, 9000),
        )
        assert path is not None
        assert len(path) > 0

    def test_disruption_blocks_path(self):
        router = SelfHealingRouter(
            volume_nm=(5000, 5000, 5000),
            grid_resolution_nm=1000,
        )
        # Block the entire middle
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    router._availability[x, y, z] = 0.0

        path = router.find_path(
            start_nm=(0, 0, 0),
            end_nm=(4000, 4000, 4000),
        )
        assert path is None

    def test_disrupted_fraction(self):
        router = SelfHealingRouter(
            volume_nm=(5000, 5000, 5000),
            grid_resolution_nm=1000,
        )
        assert router.disrupted_fraction == 0.0

        event = CosmicRayEvent(
            position_nm=(2500, 2500, 2500),
            energy_MeV=500,
            affected_radius_nm=2000,
            coherence_loss=0.9,
        )
        router.mark_disruption(event)
        assert router.disrupted_fraction > 0


class TestPhaseStabilityAnalyser:
    def test_phase_sensitivity(self):
        analyser = PhaseStabilityAnalyser()
        s = analyser.phase_sensitivity()
        assert s > 0

    def test_no_vibration_no_jitter(self):
        analyser = PhaseStabilityAnalyser()
        profile = VibrationProfile(
            frequencies_Hz=np.array([100.0]),
            amplitudes_nm=np.array([0.0]),
        )
        jitter = analyser.compute_phase_jitter(profile)
        assert jitter == pytest.approx(0.0)

    def test_jitter_increases_with_amplitude(self):
        analyser = PhaseStabilityAnalyser()
        p1 = VibrationProfile(np.array([100.0]), np.array([0.1]))
        p2 = VibrationProfile(np.array([100.0]), np.array([1.0]))
        assert analyser.compute_phase_jitter(p2) > analyser.compute_phase_jitter(p1)

    def test_full_analysis(self):
        analyser = PhaseStabilityAnalyser()
        profile = VibrationProfile(
            frequencies_Hz=np.array([50.0, 120.0, 500.0]),
            amplitudes_nm=np.array([0.5, 0.2, 0.1]),
        )
        report = analyser.analyse(profile)
        assert isinstance(report, StabilityReport)
        assert report.rms_phase_jitter_rad > 0
        assert report.coherence_length_nm > 0
        assert report.max_tolerable_amplitude_nm > 0

    def test_low_vibration_no_isolation_needed(self):
        analyser = PhaseStabilityAnalyser()
        profile = VibrationProfile(
            frequencies_Hz=np.array([100.0]),
            amplitudes_nm=np.array([0.001]),  # sub-pm
        )
        report = analyser.analyse(profile)
        assert report.isolation_required is False

    def test_max_tolerable_amplitude(self):
        analyser = PhaseStabilityAnalyser()
        max_amp = analyser.max_tolerable_amplitude(target_ber=0.01)
        assert max_amp > 0
