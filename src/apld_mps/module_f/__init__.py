"""Module F: Thermodynamic Feedback — the 'Aethel-Kreislauf'.

Couples heat generation (Landauer principle), thermo-optic refractive index
shifts, thermal transport in the sapphire lattice, and near-field
thermophotovoltaic (TPV) energy recycling into a closed-loop model unique
to the Aethel V1 architecture.
"""

from .heat_optics import ThermoOpticSolver, ThermoOpticConfig
from .tpv_recycler import TPVRecycler, TPVConfig, StabilityResult

__all__ = [
    "ThermoOpticSolver",
    "ThermoOpticConfig",
    "TPVRecycler",
    "TPVConfig",
    "StabilityResult",
]
