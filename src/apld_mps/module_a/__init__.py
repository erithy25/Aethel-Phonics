"""Module A: Material & Geometry Editor — the physical foundation."""

from .substrate import SubstrateDatabase, SubstrateMaterial
from .tmd_layer import TMDMonolayer, ExcitonProperties
from .geometry import (
    WaveguideGeometry,
    Channel,
    YSplitter,
    RingResonator,
    InteractionZone,
)

__all__ = [
    "SubstrateDatabase",
    "SubstrateMaterial",
    "TMDMonolayer",
    "ExcitonProperties",
    "WaveguideGeometry",
    "Channel",
    "YSplitter",
    "RingResonator",
    "InteractionZone",
]
