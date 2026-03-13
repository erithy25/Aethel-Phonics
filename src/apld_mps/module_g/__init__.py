"""Module G: Quantum-Cosmic Stress Test Suite.

Simulates external disturbances — cosmic ray impacts on optical coherence,
mechanical vibrations affecting holographic addressing — and develops
self-healing algorithms for real-time data-path rerouting.
"""

from .cosmic_ray import CosmicRaySimulator, CosmicRayEvent, SelfHealingRouter
from .phase_stability import PhaseStabilityAnalyser, VibrationProfile, StabilityReport

__all__ = [
    "CosmicRaySimulator",
    "CosmicRayEvent",
    "SelfHealingRouter",
    "PhaseStabilityAnalyser",
    "VibrationProfile",
    "StabilityReport",
]
