"""Tests for Module F: Thermodynamic Feedback (Aethel-Kreislauf)."""

import numpy as np
import pytest

from apld_mps.module_f import ThermoOpticSolver, ThermoOpticConfig, TPVRecycler, TPVConfig, StabilityResult


class TestThermoOpticSolver:
    def test_initialise(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
        )
        solver = ThermoOpticSolver(cfg)
        solver.initialise()
        assert solver.temperature.shape == cfg.grid_shape

    def test_base_temperature(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
        )
        solver = ThermoOpticSolver(cfg)
        solver.initialise()
        assert np.allclose(solver.temperature, 300.0)
        assert np.allclose(solver.delta_T, 0.0)
        assert np.allclose(solver.delta_n, 0.0)

    def test_heat_source_raises_temperature(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
            dt_ps=100.0,
        )
        solver = ThermoOpticSolver(cfg)
        solver.initialise()
        intensity = np.ones(cfg.grid_shape) * 1e12  # high intensity
        solver.set_heat_from_intensity(intensity)
        solver.advance(100)
        assert solver.peak_temperature_K > 300.0

    def test_landauer_heat(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
            dt_ps=100.0,
        )
        solver = ThermoOpticSolver(cfg)
        solver.initialise()
        solver.add_landauer_heat(n_operations=1_000_000)
        solver.advance(100)
        assert solver.peak_temperature_K > 300.0

    def test_delta_n_proportional_to_delta_T(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
        )
        solver = ThermoOpticSolver(cfg)
        solver.initialise()
        # Manually set a temperature rise
        solver._temperature += 10.0
        assert solver.delta_n.max() == pytest.approx(cfg.dn_dT * 10.0)

    def test_eps_r_correction(self):
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
        )
        solver = ThermoOpticSolver(cfg)
        solver.initialise()
        solver._temperature += 5.0
        correction = solver.eps_r_correction(base_n=1.76)
        expected = 2.0 * 1.76 * cfg.dn_dT * 5.0
        assert correction.max() == pytest.approx(expected)


class TestTPVRecycler:
    def test_radiated_power_increases_with_temperature(self):
        tpv = TPVRecycler(TPVConfig())
        P_low = tpv.radiated_power(310.0)
        P_high = tpv.radiated_power(400.0)
        assert P_high > P_low > 0

    def test_tpv_power_positive(self):
        tpv = TPVRecycler(TPVConfig())
        P = tpv.tpv_electrical_power(350.0)
        assert P >= 0

    def test_equilibrium_temperature(self):
        tpv = TPVRecycler(TPVConfig())
        T_eq = tpv.equilibrium_temperature(10.0)
        assert T_eq > 300.0  # above ambient

    def test_dissipated_power(self):
        tpv = TPVRecycler(TPVConfig(energy_per_operation_aJ=1.0))
        P = tpv.dissipated_power(100.0)  # 100 GHz
        assert P == pytest.approx(100e9 * 1e-18)

    def test_max_clock_rate_analysis(self):
        tpv = TPVRecycler(TPVConfig())
        result = tpv.analyse_max_clock_rate(
            max_temperature_K=350.0,
            search_resolution_GHz=1.0,
        )
        assert isinstance(result, StabilityResult)
        assert result.max_clock_rate_GHz > 0
        assert result.equilibrium_temperature_K <= 350.0
        assert result.recycling_efficiency >= 0
