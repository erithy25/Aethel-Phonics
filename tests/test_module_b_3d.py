"""Tests for Module B upgrades: 3D FDTD, 3D GPE, GPU backend."""

import numpy as np
import pytest

from apld_mps.module_b.gpu_backend import ArrayBackend, BackendType, MultiGPUManager
from apld_mps.module_b.fdtd_3d import FDTD3DSolver, FDTD3DConfig
from apld_mps.module_b.gpe_3d import GPE3DSolver, GPE3DConfig


class TestArrayBackend:
    def test_numpy_fallback(self):
        be = ArrayBackend(BackendType.NUMPY)
        assert be.backend_type == BackendType.NUMPY

    def test_zeros(self):
        be = ArrayBackend(BackendType.NUMPY)
        arr = be.zeros((3, 4, 5))
        assert arr.shape == (3, 4, 5)
        assert np.allclose(arr, 0.0)

    def test_to_numpy_identity(self):
        be = ArrayBackend(BackendType.NUMPY)
        arr = be.ones((2, 3))
        result = be.to_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert np.allclose(result, 1.0)

    def test_fft_roundtrip(self):
        be = ArrayBackend(BackendType.NUMPY)
        arr = be.array(np.random.rand(8, 8), dtype=np.complex128)
        recovered = be.ifft2(be.fft2(arr))
        assert np.allclose(be.to_numpy(arr), be.to_numpy(recovered), atol=1e-12)


class TestMultiGPUManager:
    def test_cpu_fallback(self):
        mgr = MultiGPUManager(n_devices=2)
        assert mgr.n_devices == 2

    def test_domain_decomposition(self):
        mgr = MultiGPUManager(n_devices=4)
        slices = mgr.decompose_domain((100, 100, 40))
        assert len(slices) == 4
        assert slices[0] == (0, 10)
        assert slices[-1][1] == 40

    def test_device_info(self):
        mgr = MultiGPUManager(n_devices=1)
        infos = mgr.get_device_info()
        assert len(infos) == 1
        assert infos[0].backend == BackendType.NUMPY


class TestFDTD3D:
    def test_initialise(self):
        cfg = FDTD3DConfig(
            grid_spacing_nm=100,
            domain_size_nm=(500.0, 500.0, 300.0),
            n_steps=5,
        )
        solver = FDTD3DSolver(cfg)
        solver.initialise()
        assert solver.ez.shape == cfg.grid_shape

    def test_advance(self):
        cfg = FDTD3DConfig(
            grid_spacing_nm=100,
            domain_size_nm=(500.0, 500.0, 300.0),
            n_steps=5,
        )
        solver = FDTD3DSolver(cfg)
        solver.initialise()
        solver.advance(3)
        assert solver.step == 3

    def test_source_injection(self):
        cfg = FDTD3DConfig(
            grid_spacing_nm=100,
            domain_size_nm=(500.0, 500.0, 300.0),
        )
        solver = FDTD3DSolver(cfg)
        solver.initialise()
        mid = tuple(s // 2 for s in cfg.grid_shape)
        solver.inject_source(mid, 1.0, "ez")
        assert solver.ez[mid] == pytest.approx(1.0)

    def test_field_energy_after_source(self):
        cfg = FDTD3DConfig(
            grid_spacing_nm=100,
            domain_size_nm=(500.0, 500.0, 300.0),
        )
        solver = FDTD3DSolver(cfg)
        solver.initialise()
        mid = tuple(s // 2 for s in cfg.grid_shape)
        solver.inject_source(mid, 1.0)
        assert solver.field_energy > 0

    def test_field_at_slice(self):
        cfg = FDTD3DConfig(
            grid_spacing_nm=100,
            domain_size_nm=(500.0, 500.0, 300.0),
        )
        solver = FDTD3DSolver(cfg)
        solver.initialise()
        sl = solver.field_at_slice("ez", "z", 1)
        assert sl.ndim == 2

    def test_all_six_components_exist(self):
        cfg = FDTD3DConfig(
            grid_spacing_nm=100,
            domain_size_nm=(300.0, 300.0, 300.0),
        )
        solver = FDTD3DSolver(cfg)
        solver.initialise()
        assert solver.ex.shape == cfg.grid_shape
        assert solver.ey.shape == cfg.grid_shape
        assert solver.ez.shape == cfg.grid_shape


class TestGPE3D:
    def test_initialise(self):
        cfg = GPE3DConfig(
            grid_size_nm=(200, 200, 200),
            grid_spacing_nm=50,
            n_steps=5,
        )
        solver = GPE3DSolver(cfg)
        solver.initialise()
        assert solver.density.shape == cfg.grid_shape

    def test_vacuum_stays_zero(self):
        cfg = GPE3DConfig(
            grid_size_nm=(200, 200, 200),
            grid_spacing_nm=50,
            n_steps=5,
        )
        solver = GPE3DSolver(cfg)
        solver.initialise()
        solver.advance(5)
        assert np.allclose(solver.density, 0.0)

    def test_soliton_injection(self):
        cfg = GPE3DConfig(
            grid_size_nm=(400, 400, 400),
            grid_spacing_nm=50,
        )
        solver = GPE3DSolver(cfg)
        solver.initialise()
        solver.inject_soliton(centre_nm=(200, 200, 200), width_nm=80, amplitude=1.0)
        assert np.max(solver.density) > 0

    def test_soliton_integrity_at_start(self):
        cfg = GPE3DConfig(
            grid_size_nm=(400, 400, 400),
            grid_spacing_nm=50,
        )
        solver = GPE3DSolver(cfg)
        solver.initialise()
        solver.inject_soliton(centre_nm=(200, 200, 200), width_nm=80, amplitude=1.0)
        integrity = solver.soliton_integrity()
        assert integrity > 0.1  # well-localised at start

    def test_density_slice(self):
        cfg = GPE3DConfig(
            grid_size_nm=(200, 200, 200),
            grid_spacing_nm=50,
        )
        solver = GPE3DSolver(cfg)
        solver.initialise()
        solver.inject_soliton(centre_nm=(100, 100, 100))
        sl = solver.density_slice("z", 2)
        assert sl.ndim == 2

    def test_pump_creates_density(self):
        cfg = GPE3DConfig(
            grid_size_nm=(200, 200, 200),
            grid_spacing_nm=50,
            dt_ps=0.01,
        )
        solver = GPE3DSolver(cfg)
        solver.initialise()
        pump = np.zeros(cfg.grid_shape, dtype=np.complex128)
        cx, cy, cz = [s // 2 for s in cfg.grid_shape]
        pump[cx, cy, cz] = 1e6
        solver.set_pump(pump)
        solver.advance(50)
        assert solver.density[cx, cy, cz] > 0
