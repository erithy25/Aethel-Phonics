"""Tests for Module B: Multiphysics Simulation Core."""

import numpy as np
import pytest

from apld_mps.module_b import FDTDSolver, FDTDConfig, CoupledOscillatorSolver, GrossPitaevskiiSolver, GPEConfig


class TestFDTD:
    def test_initialise_2d(self):
        cfg = FDTDConfig(grid_spacing_nm=50, domain_size_nm=(1000.0, 500.0), n_steps=10)
        solver = FDTDSolver(cfg)
        solver.initialise()
        assert solver.ez.shape == cfg.grid_shape

    def test_advance_increases_step(self):
        cfg = FDTDConfig(grid_spacing_nm=50, domain_size_nm=(500.0, 500.0), n_steps=10)
        solver = FDTDSolver(cfg)
        solver.initialise()
        solver.advance(5)
        assert solver.step == 5

    def test_source_injection(self):
        cfg = FDTDConfig(grid_spacing_nm=50, domain_size_nm=(500.0, 500.0))
        solver = FDTDSolver(cfg)
        solver.initialise()
        mid = (cfg.grid_shape[0] // 2, cfg.grid_shape[1] // 2)
        solver.inject_source(mid, 1.0)
        assert solver.ez[mid] == pytest.approx(1.0)

    def test_field_energy_after_source(self):
        cfg = FDTDConfig(grid_spacing_nm=50, domain_size_nm=(500.0, 500.0))
        solver = FDTDSolver(cfg)
        solver.initialise()
        mid = (cfg.grid_shape[0] // 2, cfg.grid_shape[1] // 2)
        solver.inject_source(mid, 1.0)
        assert solver.field_energy > 0


class TestCoupledOscillator:
    def test_strong_coupling_satisfied(self):
        solver = CoupledOscillatorSolver(
            exciton_energy_eV=1.72,
            cavity_energy_eV_at_k0=1.72,
            rabi_splitting_eV=0.030,
            cavity_linewidth_eV=0.005,
            exciton_linewidth_eV=0.010,
        )
        assert solver.check_strong_coupling() is True

    def test_strong_coupling_violated(self):
        solver = CoupledOscillatorSolver(
            exciton_energy_eV=1.72,
            cavity_energy_eV_at_k0=1.72,
            rabi_splitting_eV=0.005,
            cavity_linewidth_eV=0.010,
            exciton_linewidth_eV=0.010,
        )
        assert solver.check_strong_coupling() is False

    def test_dispersion_splitting(self):
        solver = CoupledOscillatorSolver(
            exciton_energy_eV=1.72,
            cavity_energy_eV_at_k0=1.72,
            rabi_splitting_eV=0.030,
            cavity_linewidth_eV=0.005,
            exciton_linewidth_eV=0.010,
        )
        result = solver.compute_dispersion(n_points=100)
        # At k=0 (resonance), splitting = Rabi splitting
        gap = result.E_upb[0] - result.E_lpb[0]
        assert gap == pytest.approx(0.030, abs=0.002)
        assert result.is_strong_coupling is True

    def test_lpb_below_upb(self):
        solver = CoupledOscillatorSolver(
            exciton_energy_eV=1.72,
            cavity_energy_eV_at_k0=1.72,
            rabi_splitting_eV=0.030,
            cavity_linewidth_eV=0.005,
            exciton_linewidth_eV=0.010,
        )
        result = solver.compute_dispersion()
        assert np.all(result.E_lpb < result.E_upb)


class TestGPE:
    def test_initialise(self):
        cfg = GPEConfig(grid_size_nm=(500, 500), grid_spacing_nm=50, n_steps=10)
        solver = GrossPitaevskiiSolver(cfg)
        solver.initialise()
        assert solver.density.shape == cfg.grid_shape

    def test_vacuum_stays_zero(self):
        cfg = GPEConfig(grid_size_nm=(500, 500), grid_spacing_nm=50, n_steps=10)
        solver = GrossPitaevskiiSolver(cfg)
        solver.initialise()
        solver.advance(10)
        assert np.allclose(solver.density, 0.0)

    def test_pump_creates_density(self):
        cfg = GPEConfig(grid_size_nm=(500, 500), grid_spacing_nm=50, dt_ps=0.01, n_steps=100)
        solver = GrossPitaevskiiSolver(cfg)
        solver.initialise()
        pump = np.zeros(cfg.grid_shape, dtype=np.complex128)
        cx, cy = cfg.grid_shape[0] // 2, cfg.grid_shape[1] // 2
        pump[cx, cy] = 1e6
        solver.set_pump(pump)
        solver.advance(100)
        assert solver.density[cx, cy] > 0
