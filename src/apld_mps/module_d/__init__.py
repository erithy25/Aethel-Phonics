"""Module D: Logic Compiler and Mapping Layer — the bridge to computer science.

Supports 2D flat layouts and 3D volumetric routing with RL-based optimisation.
"""

from .gate_library import GateType, LogicGate, GateLibrary
from .mapper import ChipLayout, AutoMapper
from .mapper_3d import AutoMapper3D, VolumetricLayout, PlacedGate3D, Wire3D

__all__ = [
    "GateType",
    "LogicGate",
    "GateLibrary",
    "ChipLayout",
    "AutoMapper",
    "AutoMapper3D",
    "VolumetricLayout",
    "PlacedGate3D",
    "Wire3D",
]
