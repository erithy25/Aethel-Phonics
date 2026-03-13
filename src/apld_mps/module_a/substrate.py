"""Detail 2.1 — Substrate database: optical and thermal constants.

Provides wavelength-dependent refractive index n(λ), absorption coefficient k,
thermal conductivity κ, and thermal expansion coefficient for substrates such
as sapphire (Al₂O₃).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..constants import NM_TO_M


@dataclass
class SubstrateMaterial:
    """Physical description of a substrate material.

    Attributes:
        name: Human-readable material name.
        refractive_index_fn: n(λ) where λ is in metres.
        absorption_coefficient: k [1/m] (may also be wavelength-dependent).
        thermal_conductivity: κ [W/(m·K)] at room temperature.
        thermal_expansion_coeff: α [1/K].
    """

    name: str
    refractive_index_fn: Callable[[float | np.ndarray], float | np.ndarray]
    absorption_coefficient: float
    thermal_conductivity: float
    thermal_expansion_coeff: float

    def n(self, wavelength_nm: float | np.ndarray) -> float | np.ndarray:
        """Return refractive index at given wavelength(s) in nm."""
        return self.refractive_index_fn(np.asarray(wavelength_nm) * NM_TO_M)


def _sapphire_sellmeier(wavelength_m: float | np.ndarray) -> float | np.ndarray:
    """Sellmeier equation for ordinary-ray sapphire (Al₂O₃).

    Coefficients from Malitson & Dodge (1972).
    Valid range: 200 nm – 5 µm.
    """
    lam2 = np.asarray(wavelength_m, dtype=np.float64) ** 2 * 1e12  # convert to µm²
    n_sq = (
        1.0
        + 1.4313493 * lam2 / (lam2 - 0.0726631**2)
        + 0.6505455 * lam2 / (lam2 - 0.1193242**2)
        + 5.3414021 * lam2 / (lam2 - 18.028251**2)
    )
    return np.sqrt(n_sq)


class SubstrateDatabase:
    """Registry of available substrate materials."""

    def __init__(self) -> None:
        self._materials: dict[str, SubstrateMaterial] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            SubstrateMaterial(
                name="Sapphire (Al₂O₃)",
                refractive_index_fn=_sapphire_sellmeier,
                absorption_coefficient=0.01,  # very low at visible/NIR
                thermal_conductivity=35.0,  # W/(m·K) at 300 K
                thermal_expansion_coeff=5.0e-6,  # 1/K
            )
        )

    def register(self, material: SubstrateMaterial) -> None:
        self._materials[material.name] = material

    def get(self, name: str) -> SubstrateMaterial:
        if name not in self._materials:
            available = ", ".join(self._materials)
            raise KeyError(
                f"Substrate '{name}' not found. Available: {available}"
            )
        return self._materials[name]

    def list_materials(self) -> list[str]:
        return list(self._materials.keys())
