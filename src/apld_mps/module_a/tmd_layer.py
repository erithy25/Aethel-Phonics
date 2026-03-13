"""Detail 2.2 — Active layer specification for 2D transition-metal dichalcogenides.

Full parametrisation of monolayer TMD materials (WSe₂, MoSe₂, etc.) including
exciton binding energy, oscillator strength, resonance frequency, and decay rates.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import EV_TO_J, HBAR


@dataclass
class ExcitonProperties:
    """Exciton parameters of a 2D-TMD monolayer.

    Attributes:
        binding_energy_eV: Exciton binding energy [eV].
        oscillator_strength: Dimensionless oscillator strength f.
        resonance_energy_eV: Exciton resonance energy [eV] (= ℏω_exc).
        effective_mass_ratio: Exciton effective mass as fraction of free electron mass.
    """

    binding_energy_eV: float
    oscillator_strength: float
    resonance_energy_eV: float
    effective_mass_ratio: float

    @property
    def resonance_freq_rad(self) -> float:
        """Angular resonance frequency ω_exc [rad/s]."""
        return self.resonance_energy_eV * EV_TO_J / HBAR


@dataclass
class TMDMonolayer:
    """Complete parametrisation of a monolayer TMD active layer.

    Attributes:
        name: Material identifier (e.g. "WSe2").
        exciton: Exciton properties.
        gamma_nr: Non-radiative decay rate [rad/s].
        gamma_r: Radiative decay rate [rad/s].
        thickness_nm: Effective monolayer thickness [nm].
    """

    name: str
    exciton: ExcitonProperties
    gamma_nr: float
    gamma_r: float
    thickness_nm: float = 0.7  # typical TMD monolayer

    @property
    def total_linewidth(self) -> float:
        """Total exciton linewidth γ_exc = γ_nr + γ_r [rad/s]."""
        return self.gamma_nr + self.gamma_r


# ---------------------------------------------------------------------------
# Pre-defined monolayer materials
# ---------------------------------------------------------------------------

WSe2 = TMDMonolayer(
    name="WSe2",
    exciton=ExcitonProperties(
        binding_energy_eV=0.37,
        oscillator_strength=0.15,
        resonance_energy_eV=1.72,
        effective_mass_ratio=0.29,
    ),
    gamma_nr=1.0e12,  # ~1 ps⁻¹ non-radiative
    gamma_r=2.0e11,  # ~5 ps radiative lifetime
)

MoSe2 = TMDMonolayer(
    name="MoSe2",
    exciton=ExcitonProperties(
        binding_energy_eV=0.47,
        oscillator_strength=0.14,
        resonance_energy_eV=1.66,
        effective_mass_ratio=0.35,
    ),
    gamma_nr=1.2e12,
    gamma_r=1.8e11,
)
