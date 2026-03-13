"""Detail 3.3 — Nonlinear polariton interaction solver: Gross-Pitaevskii equation.

Solves the driven-dissipative Gross-Pitaevskii equation (GPE) for the polariton
condensate wave-function ψ(r, t), modelling polariton–polariton interactions,
optical bistability, and the nonlinear switching that underpins logic gates.

    iℏ ∂ψ/∂t = [-ℏ²∇²/(2m_LP) + V(r) + g_pp|ψ|² + iℏ(R·n_R - γ_LP)/2] ψ
                + F_pump(r, t)

where g_pp is the polariton–polariton interaction strength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import HBAR, M_ELECTRON, EV_TO_J, NM_TO_M


@dataclass
class GPEConfig:
    """Configuration for the Gross-Pitaevskii solver.

    Attributes:
        grid_size_nm: (Nx, Ny) spatial domain [nm].
        grid_spacing_nm: Δx = Δy [nm].
        dt_ps: Time step [ps].
        n_steps: Number of time steps.
        m_lp_ratio: Lower-polariton effective mass as fraction of m_e.
        g_pp_eV_um2: Polariton–polariton interaction strength [eV·µm²].
        gamma_lp_ps: Polariton decay rate [1/ps].
    """

    grid_size_nm: tuple[float, float] = (5000.0, 5000.0)
    grid_spacing_nm: float = 20.0
    dt_ps: float = 0.01
    n_steps: int = 5000
    m_lp_ratio: float = 5e-5
    g_pp_eV_um2: float = 0.002  # typical for TMD polaritons
    gamma_lp_ps: float = 0.1  # ~10 ps lifetime

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (
            int(np.ceil(self.grid_size_nm[0] / self.grid_spacing_nm)),
            int(np.ceil(self.grid_size_nm[1] / self.grid_spacing_nm)),
        )


class GrossPitaevskiiSolver:
    """Driven-dissipative GPE solver for polariton condensate dynamics.

    Uses a split-step Fourier method for efficient propagation:
    1. Half-step nonlinear + potential in real space.
    2. Full-step kinetic energy in Fourier space.
    3. Half-step nonlinear + potential in real space.
    """

    def __init__(self, config: GPEConfig) -> None:
        self.cfg = config
        self._psi: np.ndarray | None = None
        self._potential: np.ndarray | None = None
        self._pump: np.ndarray | None = None
        self._k2: np.ndarray | None = None
        self.time_ps: float = 0.0
        self._initialised = False

    def initialise(
        self,
        potential_eV: np.ndarray | None = None,
    ) -> None:
        """Set up the simulation grid and k-space arrays."""
        nx, ny = self.cfg.grid_shape
        dx = self.cfg.grid_spacing_nm * NM_TO_M

        self._psi = np.zeros((nx, ny), dtype=np.complex128)

        if potential_eV is not None:
            assert potential_eV.shape == (nx, ny)
            self._potential = potential_eV
        else:
            self._potential = np.zeros((nx, ny), dtype=np.float64)

        self._pump = np.zeros((nx, ny), dtype=np.complex128)

        # k-space grid for kinetic energy operator
        kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=dx) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        self._k2 = KX**2 + KY**2

        self.time_ps = 0.0
        self._initialised = True

    def set_pump(self, pump_field: np.ndarray) -> None:
        """Set the coherent pump field F_pump(r) [√(eV/µm²·ps)]."""
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")
        assert pump_field.shape == self.cfg.grid_shape
        self._pump = pump_field.astype(np.complex128)

    def advance(self, n_steps: int | None = None) -> None:
        """Propagate ψ(r,t) forward using split-step Fourier method."""
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")
        assert self._psi is not None
        assert self._potential is not None
        assert self._pump is not None
        assert self._k2 is not None

        steps = n_steps if n_steps is not None else self.cfg.n_steps
        dt = self.cfg.dt_ps * 1e-12  # to seconds
        m_lp = self.cfg.m_lp_ratio * M_ELECTRON
        gamma = self.cfg.gamma_lp_ps * 1e12  # to 1/s
        # g_pp: convert from eV·µm² to J·m²
        g_pp = self.cfg.g_pp_eV_um2 * EV_TO_J * 1e-12

        # Kinetic propagator (full step)
        kinetic_phase = np.exp(-1j * HBAR * self._k2 * dt / (2.0 * m_lp))

        for _ in range(steps):
            density = np.abs(self._psi) ** 2

            # Nonlinear + potential + loss/gain (half step)
            V_eff = (self._potential * EV_TO_J + g_pp * density) / HBAR
            nonlin_phase_half = np.exp(
                -1j * V_eff * dt / 2.0 - gamma * dt / 4.0
            )
            self._psi = nonlin_phase_half * self._psi

            # Kinetic step in Fourier space
            psi_k = np.fft.fft2(self._psi)
            psi_k *= kinetic_phase
            self._psi = np.fft.ifft2(psi_k)

            # Nonlinear + potential + loss/gain (half step) + pump
            density = np.abs(self._psi) ** 2
            V_eff = (self._potential * EV_TO_J + g_pp * density) / HBAR
            nonlin_phase_half = np.exp(
                -1j * V_eff * dt / 2.0 - gamma * dt / 4.0
            )
            self._psi = nonlin_phase_half * self._psi + self._pump * dt

            self.time_ps += self.cfg.dt_ps

    @property
    def psi(self) -> np.ndarray:
        if self._psi is None:
            raise RuntimeError("Solver not initialised.")
        return self._psi

    @property
    def density(self) -> np.ndarray:
        """Polariton density |ψ(r)|² [1/m²]."""
        return np.abs(self.psi) ** 2

    def bistability_curve(
        self,
        pump_amplitudes: np.ndarray,
        probe_point: tuple[int, int],
        steps_per_point: int = 2000,
    ) -> np.ndarray:
        """Sweep pump amplitude and record steady-state density at probe_point.

        This traces out the S-shaped optical bistability curve characteristic
        of polariton nonlinear switching.
        """
        densities = np.zeros_like(pump_amplitudes, dtype=np.float64)
        for i, amp in enumerate(pump_amplitudes):
            self.initialise(self._potential)
            pump = np.zeros(self.cfg.grid_shape, dtype=np.complex128)
            cx, cy = self.cfg.grid_shape[0] // 2, self.cfg.grid_shape[1] // 2
            pump[cx, cy] = amp
            self.set_pump(pump)
            self.advance(steps_per_point)
            densities[i] = self.density[probe_point]
        return densities
