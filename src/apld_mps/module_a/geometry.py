"""Detail 2.3 — 2D/3D CAD geometry editor for waveguide structures.

Provides primitives for drawing nanometre-precision waveguide channels,
Y-splitters, ring resonators, and interaction zones, as well as composite
parametrised gate geometries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Point:
    """2D/3D coordinate in nanometres."""

    x: float
    y: float
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def distance_to(self, other: Point) -> float:
        return float(np.linalg.norm(self.to_array() - other.to_array()))


class ComponentType(Enum):
    CHANNEL = auto()
    Y_SPLITTER = auto()
    RING_RESONATOR = auto()
    INTERACTION_ZONE = auto()


@dataclass
class Channel:
    """Straight or curved waveguide channel with nanometre precision.

    Attributes:
        points: Ordered sequence of vertices defining the channel path.
        width_nm: Channel width [nm].
        depth_nm: Etch depth / confinement depth [nm].
    """

    points: list[Point]
    width_nm: float = 200.0
    depth_nm: float = 150.0
    component_type: ComponentType = field(default=ComponentType.CHANNEL, init=False)

    @property
    def length_nm(self) -> float:
        total = 0.0
        for a, b in zip(self.points[:-1], self.points[1:]):
            total += a.distance_to(b)
        return total


@dataclass
class YSplitter:
    """Y-junction beam splitter.

    Attributes:
        origin: Junction point.
        angle_deg: Full opening angle between the two output arms [degrees].
        arm_length_nm: Length of each output arm [nm].
        channel_width_nm: Width of all three arms [nm].
    """

    origin: Point
    angle_deg: float = 30.0
    arm_length_nm: float = 2000.0
    channel_width_nm: float = 200.0
    component_type: ComponentType = field(default=ComponentType.Y_SPLITTER, init=False)


@dataclass
class RingResonator:
    """Circular ring resonator side-coupled to a bus waveguide.

    Attributes:
        centre: Ring centre coordinate.
        radius_nm: Ring radius [nm].
        gap_nm: Gap between ring and bus waveguide [nm].
        ring_width_nm: Width of the ring waveguide [nm].
    """

    centre: Point
    radius_nm: float = 5000.0
    gap_nm: float = 100.0
    ring_width_nm: float = 200.0
    component_type: ComponentType = field(
        default=ComponentType.RING_RESONATOR, init=False
    )

    @property
    def circumference_nm(self) -> float:
        return 2.0 * np.pi * self.radius_nm


@dataclass
class InteractionZone:
    """Region where two waveguide channels come close enough for evanescent
    coupling / polariton–polariton interaction.

    Attributes:
        centre: Centre of the interaction region.
        length_nm: Interaction length [nm].
        separation_nm: Centre-to-centre distance between the two guides [nm].
    """

    centre: Point
    length_nm: float = 1000.0
    separation_nm: float = 150.0
    component_type: ComponentType = field(
        default=ComponentType.INTERACTION_ZONE, init=False
    )


@dataclass
class WaveguideGeometry:
    """Composite geometry consisting of multiple waveguide components.

    Acts as the top-level container that can be fed into the simulation core.
    """

    name: str = "unnamed"
    components: list[Channel | YSplitter | RingResonator | InteractionZone] = field(
        default_factory=list
    )
    grid_resolution_nm: float = 10.0  # spatial discretisation for simulation

    def add(
        self, component: Channel | YSplitter | RingResonator | InteractionZone
    ) -> None:
        self.components.append(component)

    @property
    def bounding_box(self) -> tuple[Point, Point]:
        """Axis-aligned bounding box (min_corner, max_corner) in nm."""
        xs: list[float] = []
        ys: list[float] = []
        for comp in self.components:
            if isinstance(comp, Channel):
                for p in comp.points:
                    xs.append(p.x)
                    ys.append(p.y)
            elif isinstance(comp, (YSplitter, RingResonator, InteractionZone)):
                origin = (
                    comp.origin
                    if isinstance(comp, YSplitter)
                    else comp.centre
                )
                xs.append(origin.x)
                ys.append(origin.y)
        if not xs:
            return Point(0, 0), Point(0, 0)
        return Point(min(xs), min(ys)), Point(max(xs), max(ys))

    @staticmethod
    def xor_interferometer(
        theta_deg: float = 120.0,
        arm_length_nm: float = 5000.0,
        interaction_length_nm: float = 1000.0,
    ) -> WaveguideGeometry:
        """Parametrised interference-based XOR gate geometry.

        Two input waveguides merge via a Y-junction, propagate through an
        interaction zone, and split again to produce an output whose intensity
        encodes the XOR truth table via destructive/constructive interference.
        """
        geo = WaveguideGeometry(name=f"XOR-interferometer-{theta_deg:.0f}deg")
        origin = Point(0, 0)

        # Input Y-splitter (reversed — acts as combiner)
        geo.add(
            YSplitter(
                origin=origin,
                angle_deg=theta_deg,
                arm_length_nm=arm_length_nm,
            )
        )

        # Interaction zone after the combiner
        iz_centre = Point(arm_length_nm + interaction_length_nm / 2, 0)
        geo.add(
            InteractionZone(
                centre=iz_centre,
                length_nm=interaction_length_nm,
            )
        )

        # Output Y-splitter
        out_origin = Point(arm_length_nm + interaction_length_nm, 0)
        geo.add(
            YSplitter(
                origin=out_origin,
                angle_deg=theta_deg,
                arm_length_nm=arm_length_nm,
            )
        )

        return geo
