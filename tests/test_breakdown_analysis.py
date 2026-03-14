"""Tests for coupled thermo-FDTD feedback and breakdown clock-rate analysis."""

import numpy as np
import pytest

from apld_mps.breakdown_analysis import (
    BreakdownAnalyser,
    BreakdownResult,
    CoupledConfig,
    CoupledStepResult,
    CoupledThermoFDTDSimulator,
)


class TestCoupledThermoFDTDSimulator:
    """Tests for the bidirectional FDTD ↔ thermal coupling."""

    def _small_config(self) -> CoupledConfig:
        """Create a small config for fast tests."""
        from apld_mps.module_b.fdtd_solver import FDTDConfig
        from apld_mps.module_f.heat_optics import ThermoOpticConfig

        return CoupledConfig(
            fdtd_config=FDTDConfig(
                grid_spacing_nm=100.0,
                domain_size_nm=(500.0, 500.0),
                courant_factor=0.5,
                n_steps=100,
                pml_layers=5,
            ),
            thermo_config=ThermoOpticConfig(
                grid_size_nm=(500.0, 500.0, 200.0),
                grid_spacing_nm=100.0,
                dt_ps=10.0,
            ),
            coupling_interval=10,
            n_ops_per_cycle=100_000,
            source_amplitude=1.0,
        )

    def test_initialise(self):
        cfg = self._small_config()
        sim = CoupledThermoFDTDSimulator(cfg)
        sim.initialise()
        assert sim.fdtd.step == 0
        assert sim.thermo.peak_temperature_K == pytest.approx(300.0)

    def test_run_produces_history(self):
        cfg = self._small_config()
        sim = CoupledThermoFDTDSimulator(cfg)
        sim.initialise()
        history = sim.run(3)
        assert len(history) == 3
        assert all(isinstance(s, CoupledStepResult) for s in history)

    def test_em_steps_advance(self):
        cfg = self._small_config()
        sim = CoupledThermoFDTDSimulator(cfg)
        sim.initialise()
        sim.run(5)
        assert sim.fdtd.step == 5 * cfg.coupling_interval

    def test_temperature_rises_with_coupling(self):
        cfg = self._small_config()
        cfg.n_ops_per_cycle = 10_000_000  # lots of ops → more heat
        sim = CoupledThermoFDTDSimulator(cfg)
        sim.initialise()
        sim.run(5)
        assert sim.peak_delta_T > 0.0

    def test_eps_r_updates_from_thermal_feedback(self):
        """Verify that a temperature rise in the thermo solver correctly
        propagates to the FDTD permittivity map via the coupling mechanism."""
        cfg = self._small_config()
        sim = CoupledThermoFDTDSimulator(cfg)
        sim.initialise()
        eps_r_before = sim.fdtd._eps_r.copy()

        # Manually inject a large temperature rise (simulating steady state)
        sim.thermo._temperature += 50.0  # +50 K

        # Run one cycle — the coupling code should pick up the ΔT
        sim.run(1)

        # eps_r should have increased: Δε_r = 2n·(dn/dT)·ΔT > 0
        # With ΔT~50K: Δε_r ≈ 2*1.76*1.3e-5*50 = 2.29e-3
        assert np.all(sim.fdtd._eps_r > eps_r_before)
        # Check magnitude is in the right ballpark
        delta_eps = sim.fdtd._eps_r - eps_r_before
        assert delta_eps.mean() > 1e-4

    def test_history_records_field_energy(self):
        cfg = self._small_config()
        sim = CoupledThermoFDTDSimulator(cfg)
        sim.initialise()
        history = sim.run(3)
        # All energies should be finite and non-negative
        for snap in history:
            assert np.isfinite(snap.field_energy_J)
            assert snap.field_energy_J >= 0

    def test_not_initialised_raises(self):
        cfg = self._small_config()
        sim = CoupledThermoFDTDSimulator(cfg)
        with pytest.raises(RuntimeError, match="initialise"):
            sim.run(1)


class TestBreakdownAnalyser:
    """Tests for the breakdown clock-rate sweep."""

    def test_sweep_returns_result(self):
        analyser = BreakdownAnalyser()
        result = analyser.sweep(clock_min_GHz=1.0, clock_max_GHz=500.0, n_points=50)
        assert isinstance(result, BreakdownResult)

    def test_sweep_data_lengths_match(self):
        analyser = BreakdownAnalyser()
        result = analyser.sweep(n_points=30)
        assert len(result.sweep_clock_rates_GHz) == 30
        assert len(result.sweep_delta_T_K) == 30
        assert len(result.sweep_extinction_ratio_dB) == 30
        assert len(result.sweep_phase_shift_rad) == 30

    def test_delta_T_increases_with_clock_rate(self):
        analyser = BreakdownAnalyser()
        result = analyser.sweep(clock_min_GHz=1.0, clock_max_GHz=500.0, n_points=20)
        # ΔT must be monotonically increasing with clock rate
        for i in range(1, len(result.sweep_delta_T_K)):
            assert result.sweep_delta_T_K[i] >= result.sweep_delta_T_K[i - 1]

    def test_extinction_ratio_decreases_with_clock_rate(self):
        analyser = BreakdownAnalyser()
        result = analyser.sweep(clock_min_GHz=1.0, clock_max_GHz=500.0, n_points=20)
        # ER must generally decrease (allow small numerical noise)
        er = result.sweep_extinction_ratio_dB
        # Overall trend: first ER should be higher than last
        assert er[0] > er[-1]

    def test_breakdown_clock_positive(self):
        analyser = BreakdownAnalyser()
        result = analyser.sweep(clock_min_GHz=1.0, clock_max_GHz=500.0, n_points=100)
        assert result.breakdown_clock_GHz > 0
        assert result.max_safe_clock_GHz > 0
        assert result.max_safe_clock_GHz <= result.breakdown_clock_GHz

    def test_breakdown_delta_n_positive(self):
        analyser = BreakdownAnalyser()
        result = analyser.sweep(n_points=50)
        assert result.breakdown_delta_n > 0
        assert result.breakdown_delta_T_K > 0

    def test_phase_shift_at_breakdown_significant(self):
        analyser = BreakdownAnalyser()
        result = analyser.sweep(n_points=100)
        # At breakdown, phase shift should be significant (> 0.1 rad)
        assert result.breakdown_phase_shift_rad > 0.1

    def test_higher_ops_lowers_breakdown_clock(self):
        """More operations per cycle → more heat → lower breakdown clock."""
        result_low = BreakdownAnalyser(n_ops_per_cycle=1_000_000_000).sweep(
            clock_max_GHz=1000.0, n_points=50
        )
        result_high = BreakdownAnalyser(n_ops_per_cycle=100_000_000_000).sweep(
            clock_max_GHz=1000.0, n_points=50
        )
        assert result_high.breakdown_clock_GHz < result_low.breakdown_clock_GHz

    def test_larger_interaction_length_lowers_breakdown(self):
        """Longer interaction zone → more phase accumulation → lower breakdown."""
        result_short = BreakdownAnalyser(
            interaction_length_nm=10_000.0,
        ).sweep(clock_max_GHz=1000.0, n_points=50)
        result_long = BreakdownAnalyser(
            interaction_length_nm=200_000.0,
        ).sweep(clock_max_GHz=1000.0, n_points=50)
        assert result_long.breakdown_clock_GHz < result_short.breakdown_clock_GHz

    def test_equilibrium_delta_T_zero_at_zero_clock(self):
        analyser = BreakdownAnalyser()
        dT = analyser._equilibrium_delta_T(0.0)
        assert dT == pytest.approx(0.0, abs=0.01)

    def test_phase_shift_proportional_to_delta_n(self):
        analyser = BreakdownAnalyser(interaction_length_nm=2000.0, wavelength_nm=721.0)
        phi1 = analyser._phase_shift(1e-4)
        phi2 = analyser._phase_shift(2e-4)
        assert phi2 == pytest.approx(2.0 * phi1, rel=1e-10)

    def test_extinction_ratio_perfect_at_zero_shift(self):
        analyser = BreakdownAnalyser()
        er = analyser._extinction_ratio_dB(0.0)
        # Should be very high (near-infinite contrast)
        assert er > 100.0

    def test_extinction_ratio_zero_at_pi_over_4(self):
        analyser = BreakdownAnalyser()
        er = analyser._extinction_ratio_dB(np.pi / 4)
        # At π/4, cos²=sin²=0.5 → ER = 0 dB
        assert er == pytest.approx(0.0, abs=0.01)
