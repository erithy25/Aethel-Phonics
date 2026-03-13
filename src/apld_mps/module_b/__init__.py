"""Module B: Multiphysics Simulation Core — the mathematical engine.

Solves coupled Maxwell electrodynamics and quantum-mechanical equations for
exciton-polariton systems. Supports 2D and 3D with GPU acceleration.
"""

from .fdtd_solver import FDTDSolver, FDTDConfig
from .fdtd_3d import FDTD3DSolver, FDTD3DConfig
from .coupled_oscillator import CoupledOscillatorSolver, DispersionResult
from .gross_pitaevskii import GrossPitaevskiiSolver, GPEConfig
from .gpe_3d import GPE3DSolver, GPE3DConfig
from .gpu_backend import ArrayBackend, BackendType, MultiGPUManager

__all__ = [
    "FDTDSolver",
    "FDTDConfig",
    "FDTD3DSolver",
    "FDTD3DConfig",
    "CoupledOscillatorSolver",
    "DispersionResult",
    "GrossPitaevskiiSolver",
    "GPEConfig",
    "GPE3DSolver",
    "GPE3DConfig",
    "ArrayBackend",
    "BackendType",
    "MultiGPUManager",
]
