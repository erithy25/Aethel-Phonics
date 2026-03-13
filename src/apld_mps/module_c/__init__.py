"""Module C: Input/Output and Detection — stimulus and measurement."""

from .laser_source import LaserSource, LaserPulse, PolarisationState
from .detector import VirtualDetector, DetectorReading

__all__ = [
    "LaserSource",
    "LaserPulse",
    "PolarisationState",
    "VirtualDetector",
    "DetectorReading",
]
