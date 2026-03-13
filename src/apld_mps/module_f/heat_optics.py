"""Detail 3.1 — Coupled heat-optics simulation.

Every computation dissipates at least k_B T ln 2 per bit (Landauer principle).
This module computes:
1. Local heat generation from absorption losses and Landauer dissipation.
2. Transient heat diffusion in the sapphire lattice (3D Fourier-based solver).
3. Thermo-optic refractive-index perturbation Δn(r, t) fed back into the
   FDTD/GPE solvers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import (
    K_BOLTZMANN,
    T_ROOM,
    NM_TO_M,
)


@dataclass
class ThermoOpticConfig:
    """Configuration for the coupled heat-optics solver.

    Attributes:
        grid_size_nm: (Lx, Ly, Lz) domain matching the simulation core [nm].
        grid_spacing_nm: Spatial step [nm].
        dt_ps: Thermal time step [ps] — typically much coarser than EM step.
        thermal_conductivity: κ of sapphire [W/(m·K)].
        density: ρ of sapphire [kg/m³].
        specific_heat: c_p of sapphire [J/(kg·K)].
        dn_dT: Thermo-optic coefficient dn/dT [1/K] for sapphire.
        absorption_coefficient: α [1/m] optical absorption.
        base_temperature_K: Ambient temperature [K].
    """

    grid_size_nm: tuple[float, float, float] = (5000.0, 5000.0, 2000.0)
    grid_spacing_nm: float = 50.0
    dt_ps: float = 1.0
    thermal_conductivity: float = 35.0  # W/(m·K) sapphire at 300 K
    density: float = 3980.0  # kg/m³
    specific_heat: float = 761.0  # J/(kg·K)
    dn_dT: float = 1.3e-5  # 1/K for sapphire at visible
    absorption_coefficient: float = 0.01  # 1/m
    base_temperature_K: float = T_ROOM

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return tuple(
            int(np.ceil(s / self.grid_spacing_nm)) for s in self.grid_size_nm
        )  # type: ignore[return-value]

    @property
    def thermal_diffusivity(self) -> float:
        """α_th = κ / (ρ · c_p) [m²/s]."""
        return self.thermal_conductivity / (self.density * self.specific_heat)


class ThermoOpticSolver:
    """Coupled heat diffusion and thermo-optic feedback solver.

    Solves the 3D heat equation:
        ρ c_p ∂T/∂t = κ ∇²T + Q(r, t)

    where Q is the volumetric heat source from:
    - Optical absorption: Q_abs = α · I(r,t)
    - Landauer dissipation: Q_L = N_ops · k_B T ln 2 / V_cell

    Uses spectral (FFT-based) diffusion for efficiency.
    """

    def __init__(self, config: ThermoOpticConfig) -> None:
        self.cfg = config
        self._temperature: np.ndarray | None = None
        self._heat_source: np.ndarray | None = None
        self._k2: np.ndarray | None = None
        self.time_ps: float = 0.0
        self._initialised = False

    def initialise(self) -> None:
        """Allocate temperature field and k-space diffusion kernel."""
        nx, ny, nz = self.cfg.grid_shape
        dx = self.cfg.grid_spacing_nm * NM_TO_M

        self._temperature = np.full(
            (nx, ny, nz), self.cfg.base_temperature_K, dtype=np.float64
        )
        self._heat_source = np.zeros((nx, ny, nz), dtype=np.float64)

        # k-space for spectral diffusion
        kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=dx) * 2 * np.pi
        kz = np.fft.fftfreq(nz, d=dx) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        self._k2 = KX**2 + KY**2 + KZ**2

        self.time_ps = 0.0
        self._initialised = True

    def set_heat_from_intensity(self, intensity_field: np.ndarray) -> None:
        """Compute volumetric heat source from optical intensity [W/m²].

        Q_abs = α · I(r)
        """
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")
        # Resample intensity to thermal grid if shapes differ
        if intensity_field.shape != self.cfg.grid_shape:
            from scipy.ndimage import zoom
            factors = tuple(
                t / s for t, s in zip(self.cfg.grid_shape, intensity_field.shape)
            )
            intensity_field = zoom(intensity_field, factors, order=1)
        self._heat_source = self.cfg.absorption_coefficient * intensity_field

    def add_landauer_heat(
        self, n_operations: int, volume_nm3: float | None = None
    ) -> None:
        """Add Landauer-principle minimum heat: Q_L = N · k_B T ln 2 / V.

        Distributes uniformly across the domain if volume is not specified.
        """
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")
        dx = self.cfg.grid_spacing_nm * NM_TO_M
        if volume_nm3 is None:
            vol_m3 = np.prod(np.array(self.cfg.grid_size_nm)) * NM_TO_M**3
        else:
            vol_m3 = volume_nm3 * NM_TO_M**3

        q_landauer = (
            n_operations * K_BOLTZMANN * self.cfg.base_temperature_K * np.log(2)
        ) / vol_m3
        self._heat_source += q_landauer

    def advance(self, n_steps: int = 1) -> None:
        """Advance thermal diffusion by n_steps.

        Uses spectral method: T_k(t+dt) = T_k(t) · exp(-α_th k² dt) + Q_k dt/(ρ c_p)
        """
        if not self._initialised:
            raise RuntimeError("Call initialise() first.")

        dt = self.cfg.dt_ps * 1e-12
        alpha = self.cfg.thermal_diffusivity
        rho_cp = self.cfg.density * self.cfg.specific_heat

        # Diffusion decay in k-space
        diffusion_factor = np.exp(-alpha * self._k2 * dt)

        # Source term in k-space (constant over dt)
        assert self._heat_source is not None
        Q_k = np.fft.fftn(self._heat_source) * dt / rho_cp

        for _ in range(n_steps):
            T_k = np.fft.fftn(self._temperature - self.cfg.base_temperature_K)
            T_k = T_k * diffusion_factor + Q_k
            self._temperature = np.real(np.fft.ifftn(T_k)) + self.cfg.base_temperature_K
            self.time_ps += self.cfg.dt_ps

    @property
    def temperature(self) -> np.ndarray:
        if self._temperature is None:
            raise RuntimeError("Solver not initialised.")
        return self._temperature

    @property
    def delta_T(self) -> np.ndarray:
        """Temperature rise above ambient [K]."""
        return self.temperature - self.cfg.base_temperature_K

    @property
    def delta_n(self) -> np.ndarray:
        """Thermo-optic refractive index perturbation Δn(r).

        Δn = (dn/dT) · ΔT
        """
        return self.cfg.dn_dT * self.delta_T

    @property
    def peak_temperature_K(self) -> float:
        return float(np.max(self.temperature))

    @property
    def mean_temperature_K(self) -> float:
        return float(np.mean(self.temperature))

    def eps_r_correction(self, base_n: float) -> np.ndarray:
        """Compute ε_r correction array to feed back into FDTD.

        ε_r(r) = (n + Δn(r))² ≈ n² + 2n·Δn
        Returns the additive correction 2n·Δn to add to the base ε_r.
        """
        return 2.0 * base_n * self.delta_n
