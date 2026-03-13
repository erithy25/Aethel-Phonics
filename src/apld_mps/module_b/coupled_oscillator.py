"""Detail 3.2 — Quantum-optical solver: coupled oscillator model for strong coupling.

Models the photon field and exciton polarisation as two coupled harmonic
oscillators to compute polariton dispersion relations (LPB / UPB) and verify
the strong-coupling condition Ω_R > (γ_cav + γ_exc) / 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import HBAR, EV_TO_J


@dataclass
class DispersionResult:
    """Result of a polariton dispersion calculation.

    Attributes:
        k: Wave-vector array [1/m].
        E_cav: Bare cavity photon energy E_cav(k) [eV].
        E_exc: Bare exciton energy (constant) [eV].
        E_lpb: Lower Polariton Branch energy [eV].
        E_upb: Upper Polariton Branch energy [eV].
        rabi_splitting_eV: Vacuum Rabi splitting Ω_R [eV].
        is_strong_coupling: Whether the strong-coupling condition is satisfied.
    """

    k: np.ndarray
    E_cav: np.ndarray
    E_exc: float
    E_lpb: np.ndarray
    E_upb: np.ndarray
    rabi_splitting_eV: float
    is_strong_coupling: bool


class CoupledOscillatorSolver:
    """Solve the 2×2 coupled-oscillator Hamiltonian for exciton–polaritons.

    The Hamiltonian (at each k-point) is:

        H = | E_cav(k)     g    |
            |   g        E_exc  |

    with coupling strength g = ℏΩ_R / 2.

    Eigenvalues give the LPB and UPB dispersions.
    """

    def __init__(
        self,
        exciton_energy_eV: float,
        cavity_energy_eV_at_k0: float,
        rabi_splitting_eV: float,
        cavity_linewidth_eV: float,
        exciton_linewidth_eV: float,
        effective_cavity_mass_ratio: float = 1e-5,
    ) -> None:
        self.E_exc = exciton_energy_eV
        self.E_cav_0 = cavity_energy_eV_at_k0
        self.omega_R_eV = rabi_splitting_eV
        self.gamma_cav_eV = cavity_linewidth_eV
        self.gamma_exc_eV = exciton_linewidth_eV
        self.m_cav_ratio = effective_cavity_mass_ratio

    def check_strong_coupling(self) -> bool:
        """Verify Ω_R > (γ_cav + γ_exc) / 2."""
        return self.omega_R_eV > (self.gamma_cav_eV + self.gamma_exc_eV) / 2.0

    def cavity_dispersion(self, k: np.ndarray) -> np.ndarray:
        """Bare cavity photon dispersion E_cav(k) [eV].

        Parabolic approximation: E_cav(k) = E_cav(0) + ℏ²k²/(2 m_cav).
        """
        from ..constants import M_ELECTRON

        m_cav = self.m_cav_ratio * M_ELECTRON
        E_kinetic_J = (HBAR * k) ** 2 / (2.0 * m_cav)
        return self.E_cav_0 + E_kinetic_J / EV_TO_J

    def compute_dispersion(
        self, k_max: float = 1e7, n_points: int = 500
    ) -> DispersionResult:
        """Diagonalise the coupled-oscillator Hamiltonian along k.

        Args:
            k_max: Maximum wave-vector [1/m].
            n_points: Number of k-points.

        Returns:
            DispersionResult with LPB, UPB, and strong-coupling flag.
        """
        k = np.linspace(0, k_max, n_points)
        E_cav = self.cavity_dispersion(k)
        g = self.omega_R_eV / 2.0  # coupling in eV

        # Eigenvalues of 2×2 Hamiltonian
        avg = (E_cav + self.E_exc) / 2.0
        delta = E_cav - self.E_exc
        splitting = np.sqrt(delta**2 + 4.0 * g**2)

        E_upb = avg + splitting / 2.0
        E_lpb = avg - splitting / 2.0

        return DispersionResult(
            k=k,
            E_cav=E_cav,
            E_exc=self.E_exc,
            E_lpb=E_lpb,
            E_upb=E_upb,
            rabi_splitting_eV=self.omega_R_eV,
            is_strong_coupling=self.check_strong_coupling(),
        )
