"""Module B: Multiphysics Simulation Core — the mathematical engine.

Solves coupled Maxwell electrodynamics and quantum-mechanical equations for
exciton-polariton systems.
"""

from .fdtd_solver import FDTDSolver, FDTDConfig
from .coupled_oscillator import CoupledOscillatorSolver, DispersionResult
from .gross_pitaevskii import GrossPitaevskiiSolver, GPEConfig

__all__ = [
    "FDTDSolver",
    "FDTDConfig",
    "CoupledOscillatorSolver",
    "DispersionResult",
    "GrossPitaevskiiSolver",
    "GPEConfig",
]
