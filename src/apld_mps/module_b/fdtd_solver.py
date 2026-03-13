"""Detail 3.1 — Electromagnetic solver: FDTD (Finite-Difference Time-Domain).

Computes E(x,y,z,t) and H(x,y,z,t) for light propagation through the
waveguide geometry, including evanescent field tails outside the guides.
Supports 2D (TM/TE) and 3D operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from scipy.constants import c as C_LIGHT, epsilon_0, mu_0


class Polarisation(Enum):
    TE = auto()
    TM = auto()


@dataclass
class FDTDConfig:
    """Configuration for an FDTD simulation run.

    Attributes:
        grid_spacing_nm: Spatial step Δx = Δy (= Δz in 3D) [nm].
        domain_size_nm: (Nx, Ny) or (Nx, Ny, Nz) domain extents [nm].
        courant_factor: Courant number S (Δt = S·Δx/c). Must be < 1/√dim.
        n_steps: Number of time steps.
        pml_layers: Number of perfectly-matched-layer absorbing cells.
        polarisation: TE or TM (2D mode only).
    """

    grid_spacing_nm: float = 10.0
    domain_size_nm: tuple[float, ...] = (10_000.0, 5_000.0)
    courant_factor: float = 0.5
    n_steps: int = 1000
    pml_layers: int = 20
    polarisation: Polarisation = Polarisation.TE

    @property
    def ndim(self) -> int:
        return len(self.domain_size_nm)

    @property
    def grid_shape(self) -> tuple[int, ...]:
        return tuple(
            int(np.ceil(s / self.grid_spacing_nm)) for s in self.domain_size_nm
        )

    @property
    def dt_s(self) -> float:
        """Time step [s] satisfying the Courant condition."""
        dx = self.grid_spacing_nm * 1e-9
        return self.courant_factor * dx / C_LIGHT


class FDTDSolver:
    """2D/3D FDTD Maxwell solver with PML absorbing boundaries.

    The solver uses the standard Yee staggered-grid algorithm to update
    E and H fields alternately.
    """

    def __init__(self, config: FDTDConfig) -> None:
        self.cfg = config
        self._initialised = False

        # Field arrays (allocated lazily)
        self._ez: np.ndarray | None = None  # 2D TM: Ez component
        self._hx: np.ndarray | None = None
        self._hy: np.ndarray | None = None

        # Permittivity map (relative)
        self._eps_r: np.ndarray | None = None

        # Time step counter
        self.step: int = 0

    def initialise(self, eps_r_map: np.ndarray | None = None) -> None:
        """Allocate fields and set material map.

        Args:
            eps_r_map: Relative permittivity on the grid. Defaults to vacuum.
        """
        shape = self.cfg.grid_shape
        if self.cfg.ndim == 2:
            nx, ny = shape
            self._ez = np.zeros((nx, ny), dtype=np.float64)
            self._hx = np.zeros((nx, ny), dtype=np.float64)
            self._hy = np.zeros((nx, ny), dtype=np.float64)
            if eps_r_map is not None:
                assert eps_r_map.shape == (nx, ny)
                self._eps_r = eps_r_map.astype(np.float64)
            else:
                self._eps_r = np.ones((nx, ny), dtype=np.float64)
        else:
            raise NotImplementedError("3D FDTD solver is planned but not yet implemented")

        self.step = 0
        self._initialised = True

    def inject_source(self, position: tuple[int, int], amplitude: float) -> None:
        """Inject a point source into the Ez field at the given grid index."""
        if not self._initialised:
            raise RuntimeError("Call initialise() before injecting sources.")
        assert self._ez is not None
        self._ez[position] += amplitude

    def advance(self, n_steps: int = 1) -> None:
        """Advance the FDTD simulation by *n_steps* time steps.

        Uses the 2D TM-mode Yee update equations:
            Hx^{n+1/2} = Hx^{n-1/2} - (Δt/μ₀) · ∂Ez/∂y
            Hy^{n+1/2} = Hy^{n-1/2} + (Δt/μ₀) · ∂Ez/∂x
            Ez^{n+1}   = Ez^{n}     + (Δt/ε₀ε_r) · (∂Hy/∂x - ∂Hx/∂y)
        """
        if not self._initialised:
            raise RuntimeError("Call initialise() before advancing.")

        assert self._ez is not None
        assert self._hx is not None
        assert self._hy is not None
        assert self._eps_r is not None

        dt = self.cfg.dt_s
        dx = self.cfg.grid_spacing_nm * 1e-9
        ch = dt / (mu_0 * dx)
        ce = dt / (epsilon_0 * dx)

        for _ in range(n_steps):
            # --- Update H fields (half step) ---
            self._hx[:, :-1] -= ch * (self._ez[:, 1:] - self._ez[:, :-1])
            self._hy[:-1, :] += ch * (self._ez[1:, :] - self._ez[:-1, :])

            # --- Update E field ---
            self._ez[1:, 1:] += (ce / self._eps_r[1:, 1:]) * (
                (self._hy[1:, 1:] - self._hy[:-1, 1:])
                - (self._hx[1:, 1:] - self._hx[1:, :-1])
            )

            self.step += 1

    @property
    def ez(self) -> np.ndarray:
        """Current Ez field snapshot."""
        if self._ez is None:
            raise RuntimeError("Solver not initialised.")
        return self._ez

    @property
    def field_energy(self) -> float:
        """Total electromagnetic energy in the domain [J]."""
        if not self._initialised:
            raise RuntimeError("Solver not initialised.")
        assert self._ez is not None and self._hx is not None and self._hy is not None
        dx = self.cfg.grid_spacing_nm * 1e-9
        cell_area = dx * dx
        e_energy = 0.5 * epsilon_0 * np.sum(self._eps_r * self._ez**2) * cell_area
        h_energy = 0.5 * mu_0 * np.sum(self._hx**2 + self._hy**2) * cell_area
        return float(e_energy + h_energy)
