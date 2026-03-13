"""Detail 2.2 — 3D nonlinear condensate dynamics: polariton bullets.

Extends the Gross-Pitaevskii equation to three spatial dimensions for
modelling polariton-bullet solitons — self-localised light packets that
propagate through the sapphire volume without dispersing.

    iℏ ∂ψ/∂t = [-ℏ²∇²/(2m_LP) + V(r) + g_pp|ψ|² + iℏ(R·n_R - γ_LP)/2] ψ
                + F_pump(r, t)

The split-step Fourier method is generalised to 3D FFTs, with GPU acceleration
via the ArrayBackend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import HBAR, M_ELECTRON, EV_TO_J, NM_TO_M
from .gpu_backend import ArrayBackend


@dataclass
class GPE3DConfig:
    """Configuration for the 3D Gross-Pitaevskii solver.

    Attributes:
        grid_size_nm: (Lx, Ly, Lz) spatial domain [nm].
        grid_spacing_nm: Δx = Δy = Δz [nm].
        dt_ps: Time step [ps].
        n_steps: Number of time steps.
        m_lp_ratio: Lower-polariton effective mass as fraction of m_e.
        g_pp_eV_um2: Polariton–polariton interaction strength [eV·µm²].
        gamma_lp_ps: Polariton decay rate [1/ps].
    """

    grid_size_nm: tuple[float, float, float] = (2000.0, 2000.0, 1000.0)
    grid_spacing_nm: float = 20.0
    dt_ps: float = 0.01
    n_steps: int = 5000
    m_lp_ratio: float = 5e-5
    g_pp_eV_um2: float = 0.002
    gamma_lp_ps: float = 0.1

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return tuple(
            int(np.ceil(s / self.grid_spacing_nm)) for s in self.grid_size_nm
        )  # type: ignore[return-value]


class GPE3DSolver:
    """3D driven-dissipative GPE solver for polariton-bullet dynamics.

    Uses 3D split-step Fourier method with GPU acceleration.
    Supports initialisation of soliton-like wave packets that propagate
    through the 3D sapphire volume.
    """

    def __init__(
        self,
        config: GPE3DConfig,
        backend: ArrayBackend | None = None,
    ) -> None:
        self.cfg = config
        self.backend = backend or ArrayBackend()
        self._psi = None
        self._potential = None
        self._pump = None
        self._k2 = None
        self.time_ps: float = 0.0
        self._initialised = False

    def initialise(self, potential_eV: np.ndarray | None = None) -> None:
        """Allocate 3D grid, k-space arrays, and kinetic propagator."""
        be = self.backend
        nx, ny, nz = self.cfg.grid_shape
        dx = self.cfg.grid_spacing_nm * NM_TO_M

        self._psi = be.zeros((nx, ny, nz), dtype=np.complex128)

        if potential_eV is not None:
            self._potential = be.array(potential_eV, dtype=np.float64)
        else:
            self._potential = be.zeros((nx, ny, nz), dtype=np.float64)

        self._pump = be.zeros((nx, ny, nz), dtype=np.complex128)

        # 3D k-space grid
        kx = be.fftfreq(nx, d=dx) * 2 * np.pi
        ky = be.fftfreq(ny, d=dx) * 2 * np.pi
        kz = be.fftfreq(nz, d=dx) * 2 * np.pi
        KX, KY, KZ = be.meshgrid(kx, ky, kz, indexing="ij")
        self._k2 = KX**2 + KY**2 + KZ**2

        self.time_ps = 0.0
        self._initialised = True

    def inject_soliton(
        self,
        centre_nm: tuple[float, float, float],
        width_nm: float = 200.0,
        amplitude: float = 1.0,
        momentum_nm_inv: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Inject a 3D Gaussian polariton-bullet wavepacket.

        The soliton is a localised wave packet with initial momentum,
        designed to propagate without dispersing when the nonlinear
        interaction balances diffraction (bright soliton condition).
        """
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")
        be = self.backend
        nx, ny, nz = self.cfg.grid_shape
        dx = self.cfg.grid_spacing_nm

        # Build coordinate grids
        x = np.arange(nx) * dx
        y = np.arange(ny) * dx
        z = np.arange(nz) * dx
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        cx, cy, cz = centre_nm
        kx, ky, kz = [k * NM_TO_M for k in momentum_nm_inv]
        w = width_nm

        # Gaussian envelope × plane wave
        r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
        envelope = amplitude * np.exp(-r2 / (2.0 * w**2))
        phase = kx * (X - cx) * NM_TO_M + ky * (Y - cy) * NM_TO_M + kz * (Z - cz) * NM_TO_M
        wavepacket = envelope * np.exp(1j * phase)

        self._psi = self._psi + be.array(wavepacket, dtype=np.complex128)

    def set_pump(self, pump_field: np.ndarray) -> None:
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")
        self._pump = self.backend.array(pump_field, dtype=np.complex128)

    def advance(self, n_steps: int | None = None) -> None:
        """Propagate ψ(r,t) forward using 3D split-step Fourier method."""
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")

        be = self.backend
        steps = n_steps if n_steps is not None else self.cfg.n_steps
        dt = self.cfg.dt_ps * 1e-12
        m_lp = self.cfg.m_lp_ratio * M_ELECTRON
        gamma = self.cfg.gamma_lp_ps * 1e12
        g_pp = self.cfg.g_pp_eV_um2 * EV_TO_J * 1e-12

        # Kinetic propagator (full step)
        kinetic_phase = be.exp(-1j * HBAR * self._k2 * dt / (2.0 * m_lp))

        for _ in range(steps):
            density = be.abs(self._psi) ** 2

            # Half-step: nonlinear + potential + dissipation
            V_eff = (self._potential * EV_TO_J + g_pp * density) / HBAR
            nonlin_half = be.exp(-1j * V_eff * dt / 2.0 - gamma * dt / 4.0)
            self._psi = nonlin_half * self._psi

            # Full-step: kinetic in k-space
            psi_k = be.fftn(self._psi)
            psi_k = psi_k * kinetic_phase
            self._psi = be.ifftn(psi_k)

            # Half-step: nonlinear + potential + dissipation + pump
            density = be.abs(self._psi) ** 2
            V_eff = (self._potential * EV_TO_J + g_pp * density) / HBAR
            nonlin_half = be.exp(-1j * V_eff * dt / 2.0 - gamma * dt / 4.0)
            self._psi = nonlin_half * self._psi + self._pump * dt

            self.time_ps += self.cfg.dt_ps

    @property
    def psi(self) -> np.ndarray:
        if self._psi is None:
            raise RuntimeError("Solver not initialised.")
        return self.backend.to_numpy(self._psi)

    @property
    def density(self) -> np.ndarray:
        """Polariton density |ψ(r)|² [1/m³]."""
        return np.abs(self.psi) ** 2

    def density_slice(self, axis: str, index: int) -> np.ndarray:
        """Extract a 2D density slice."""
        d = self.density
        if axis == "x":
            return d[index, :, :]
        elif axis == "y":
            return d[:, index, :]
        return d[:, :, index]

    def soliton_integrity(self, threshold_fraction: float = 0.5) -> float:
        """Measure how well a polariton bullet maintains its shape.

        Returns the fraction of total density contained within the region
        where density exceeds threshold_fraction × peak_density.
        A perfect soliton approaches 1.0.
        """
        d = self.density
        peak = np.max(d)
        if peak == 0:
            return 0.0
        mask = d >= threshold_fraction * peak
        return float(np.sum(d[mask]) / np.sum(d))
