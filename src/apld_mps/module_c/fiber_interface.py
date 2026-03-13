"""Detail 6.1 — Virtual hollow-core fibre interface.

Simulates the optical transition from external hollow-core glass fibres
into the internal sapphire waveguide lattice. Optimises coupling efficiency
to push reflection losses below 0.1% at the box entrance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import C_LIGHT, NM_TO_M


@dataclass
class FibreSpec:
    """Specification of an incoming hollow-core fibre.

    Attributes:
        core_diameter_um: Fibre core diameter [µm].
        numerical_aperture: NA of the fibre.
        n_core: Effective refractive index of the hollow core.
        n_cladding: Cladding index.
        wavelength_nm: Operating wavelength [nm].
    """

    core_diameter_um: float = 30.0
    numerical_aperture: float = 0.12
    n_core: float = 1.0  # hollow-core ≈ air
    n_cladding: float = 1.45
    wavelength_nm: float = 720.0


@dataclass
class CouplingResult:
    """Result of fibre-to-sapphire coupling analysis.

    Attributes:
        coupling_efficiency: Power coupling efficiency [0–1].
        reflection_loss: Fresnel reflection loss [0–1].
        mode_overlap: Spatial mode overlap integral [0–1].
        insertion_loss_dB: Total insertion loss [dB].
        optimised_taper_length_um: Optimal adiabatic taper length [µm].
    """

    coupling_efficiency: float
    reflection_loss: float
    mode_overlap: float
    insertion_loss_dB: float
    optimised_taper_length_um: float


class HollowCoreFibreInterface:
    """Model the fibre-to-sapphire coupling and optimise for < 0.1% reflection.

    Strategy:
    1. Anti-reflection coating (Fresnel loss reduction).
    2. Adiabatic mode-field taper to match sapphire waveguide mode.
    3. Spatial mode overlap integral for coupling efficiency.
    """

    def __init__(
        self,
        fibre: FibreSpec,
        sapphire_n: float = 1.76,
        waveguide_width_nm: float = 200.0,
    ) -> None:
        self.fibre = fibre
        self.n_sap = sapphire_n
        self.wg_width = waveguide_width_nm

    def fresnel_reflection(self, n1: float, n2: float) -> float:
        """Normal-incidence Fresnel reflection: R = ((n1-n2)/(n1+n2))²."""
        return ((n1 - n2) / (n1 + n2)) ** 2

    def ar_coated_reflection(self, n_coating: float | None = None) -> float:
        """Reflection with single-layer anti-reflection coating.

        Ideal AR coating: n_coat = √(n1 · n2) → R ≈ 0 at design wavelength.
        With practical coatings, residual R ~ 0.05–0.1%.
        """
        n1 = self.fibre.n_core
        n2 = self.n_sap
        if n_coating is None:
            n_coating = np.sqrt(n1 * n2)  # ideal quarter-wave
        # Residual reflection (simplified)
        r1 = self.fresnel_reflection(n1, n_coating)
        r2 = self.fresnel_reflection(n_coating, n2)
        # Destructive interference at quarter-wave → residual
        return abs(r1 - r2) ** 2

    def mode_field_diameter_um(self) -> float:
        """Gaussian MFD of the sapphire waveguide [µm]."""
        # Approximate: MFD ≈ 1.2 × waveguide width for single-mode
        return 1.2 * self.wg_width * 1e-3  # nm → µm

    def mode_overlap_integral(self) -> float:
        """Overlap between Gaussian fibre mode and waveguide mode.

        η = (2 w_f w_s / (w_f² + w_s²))² for Gaussian modes.
        """
        w_f = self.fibre.core_diameter_um / 2.0  # fibre mode radius
        w_s = self.mode_field_diameter_um() / 2.0  # sapphire mode radius
        return (2.0 * w_f * w_s / (w_f**2 + w_s**2)) ** 2

    def optimal_taper_length_um(self) -> float:
        """Adiabatic taper length to match mode fields [µm].

        L_taper ≈ (w_f - w_s) / (θ_adiabatic) where θ ~ λ / (π n w²).
        """
        w_f = self.fibre.core_diameter_um / 2.0
        w_s = self.mode_field_diameter_um() / 2.0
        lam_um = self.fibre.wavelength_nm * 1e-3
        theta = lam_um / (np.pi * self.n_sap * min(w_f, w_s) ** 2)
        return abs(w_f - w_s) / max(theta, 1e-9)

    def analyse_coupling(self) -> CouplingResult:
        """Full coupling analysis: reflection + mode overlap + insertion loss."""
        R = self.ar_coated_reflection()
        overlap = self.mode_overlap_integral()
        taper_len = self.optimal_taper_length_um()

        # Taper efficiency ~ 1 - 0.01/taper_length (longer = better)
        taper_eff = 1.0 - 0.1 / max(taper_len, 1.0)
        taper_eff = max(0.0, min(1.0, taper_eff))

        total_eff = (1.0 - R) * overlap * taper_eff
        il_dB = -10.0 * np.log10(max(total_eff, 1e-15))

        return CouplingResult(
            coupling_efficiency=total_eff,
            reflection_loss=R,
            mode_overlap=overlap,
            insertion_loss_dB=il_dB,
            optimised_taper_length_um=taper_len,
        )
