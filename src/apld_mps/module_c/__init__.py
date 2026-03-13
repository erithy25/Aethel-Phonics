"""Module C: Input/Output and Detection — stimulus, measurement, and data ingress.

Includes laser source control, virtual detectors, hollow-core fibre interface,
and petabit-scale I/O dashboard.
"""

from .laser_source import LaserSource, LaserPulse, PolarisationState
from .detector import VirtualDetector, DetectorReading
from .fiber_interface import HollowCoreFibreInterface, FibreSpec, CouplingResult
from .io_dashboard import PetabitIODashboard, ChannelStats, BottleneckReport

__all__ = [
    "LaserSource",
    "LaserPulse",
    "PolarisationState",
    "VirtualDetector",
    "DetectorReading",
    "HollowCoreFibreInterface",
    "FibreSpec",
    "CouplingResult",
    "PetabitIODashboard",
    "ChannelStats",
    "BottleneckReport",
]
