"""
APLD-MPS: Aethel Polariton Logic Device — Multiphysics Simulation Suite.

Integrated IDE and simulation platform for exciton-polariton-based logic gates
and waveguide structures at room temperature.

Modules:
    A — Material & Geometry Editor
    B — Multiphysics Simulation Core (2D/3D, GPU-accelerated)
    C — Input/Output, Detection & Data Ingress
    D — Logic Compiler & Mapping (2D flat + 3D volumetric with RL)
    E — Visualization & Performance Metrics
    F — Thermodynamic Feedback ('Aethel-Kreislauf')
    G — Quantum-Cosmic Stress Test Suite
    H — Command & Control Dashboard
    I — AI Tensor Compiler
"""

__version__ = "0.4.0"

from .breakdown_analysis import (  # noqa: E402
    BreakdownAnalyser,
    BreakdownResult,
    CoupledConfig,
    CoupledStepResult,
    CoupledThermoFDTDSimulator,
)
