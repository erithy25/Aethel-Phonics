"""Detail 5.2 — Automatic mapping: logic function → polariton chip layout.

Takes a complex logical function (e.g. "half adder"), decomposes it into
primitive polariton gates, and places them in an optimised 2D arrangement
on the sapphire chip, minimising signal delay and cross-talk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gate_library import GateType, LogicGate, GateLibrary
from ..module_a.geometry import Point


@dataclass
class PlacedGate:
    """A logic gate placed at a specific position on the chip.

    Attributes:
        gate: The logic gate definition.
        origin_nm: Bottom-left corner of the gate's bounding box on the chip [nm].
        gate_id: Unique identifier for this placed instance.
    """

    gate: LogicGate
    origin_nm: tuple[float, float]
    gate_id: str

    @property
    def centre_nm(self) -> tuple[float, float]:
        w, h = self.gate.footprint_nm
        return (self.origin_nm[0] + w / 2, self.origin_nm[1] + h / 2)


@dataclass
class Wire:
    """Optical interconnect between an output port and an input port.

    Attributes:
        from_gate_id: Source gate identifier.
        to_gate_id: Destination gate identifier.
        to_input_index: Which input of the destination gate.
        length_nm: Estimated waveguide length [nm].
    """

    from_gate_id: str
    to_gate_id: str
    to_input_index: int = 0
    length_nm: float = 0.0


@dataclass
class ChipLayout:
    """Complete chip layout: placed gates and optical interconnects.

    Attributes:
        name: Layout identifier.
        placed_gates: All placed gate instances.
        wires: Optical interconnects.
        chip_size_nm: (width, height) of the chip [nm].
    """

    name: str = "unnamed"
    placed_gates: list[PlacedGate] = field(default_factory=list)
    wires: list[Wire] = field(default_factory=list)
    chip_size_nm: tuple[float, float] = (100_000.0, 100_000.0)

    @property
    def total_wire_length_nm(self) -> float:
        return sum(w.length_nm for w in self.wires)

    @property
    def gate_count(self) -> int:
        return len(self.placed_gates)


class AutoMapper:
    """Map a logical function onto an optimised polariton chip layout.

    Strategy: greedy left-to-right placement with minimal vertical spacing.
    Gates are arranged in columns by logic depth, and wires are routed as
    straight horizontal connections where possible.
    """

    def __init__(self, library: GateLibrary | None = None) -> None:
        self.library = library or GateLibrary()
        self._spacing_nm = 2000.0  # gap between adjacent gates

    def decompose_half_adder(self) -> list[tuple[str, GateType, list[str]]]:
        """Decompose a half-adder into primitive gates.

        Half adder:
            Sum  = A XOR B
            Carry = A AND B

        Returns list of (gate_id, gate_type, [input_source_ids]).
        Input sources "A" and "B" are external.
        """
        return [
            ("xor_0", GateType.XOR, ["A", "B"]),
            ("and_0", GateType.AND, ["A", "B"]),
        ]

    def decompose_full_adder(self) -> list[tuple[str, GateType, list[str]]]:
        """Decompose a full adder (A + B + Cin) into primitive gates.

        Sum  = A XOR B XOR Cin
        Cout = (A AND B) OR ((A XOR B) AND Cin)
        """
        return [
            ("xor_0", GateType.XOR, ["A", "B"]),
            ("and_0", GateType.AND, ["A", "B"]),
            ("xor_1", GateType.XOR, ["xor_0", "Cin"]),
            ("and_1", GateType.AND, ["xor_0", "Cin"]),
            ("or_0", GateType.OR, ["and_0", "and_1"]),
        ]

    def place_and_route(
        self,
        netlist: list[tuple[str, GateType, list[str]]],
        chip_name: str = "auto-layout",
    ) -> ChipLayout:
        """Place gates and route wires on the chip.

        Uses a simple topological-sort based column assignment, then places
        gates left-to-right with vertical stacking within each column.
        """
        layout = ChipLayout(name=chip_name)

        # Determine column depth for each gate (BFS from inputs)
        depths: dict[str, int] = {}
        for gate_id, _, sources in netlist:
            max_src_depth = -1
            for src in sources:
                if src in depths:
                    max_src_depth = max(max_src_depth, depths[src])
            depths[gate_id] = max_src_depth + 1

        # Group by column
        max_depth = max(depths.values()) if depths else 0
        columns: list[list[tuple[str, GateType, list[str]]]] = [
            [] for _ in range(max_depth + 1)
        ]
        for entry in netlist:
            gate_id = entry[0]
            columns[depths[gate_id]].append(entry)

        # Place gates
        x_offset = 0.0
        gate_positions: dict[str, PlacedGate] = {}
        max_width_in_col = 0.0

        for col in columns:
            y_offset = 0.0
            max_width_in_col = 0.0
            for gate_id, gate_type, _ in col:
                gate_def = self.library.get(gate_type)
                placed = PlacedGate(
                    gate=gate_def,
                    origin_nm=(x_offset, y_offset),
                    gate_id=gate_id,
                )
                layout.placed_gates.append(placed)
                gate_positions[gate_id] = placed

                w, h = gate_def.footprint_nm
                y_offset += h + self._spacing_nm
                max_width_in_col = max(max_width_in_col, w)

            x_offset += max_width_in_col + self._spacing_nm

        # Route wires
        for gate_id, _, sources in netlist:
            dest = gate_positions[gate_id]
            for idx, src_id in enumerate(sources):
                if src_id in gate_positions:
                    src = gate_positions[src_id]
                    dx = dest.origin_nm[0] - src.origin_nm[0]
                    dy = dest.origin_nm[1] - src.origin_nm[1]
                    length = np.sqrt(dx**2 + dy**2)
                    layout.wires.append(
                        Wire(
                            from_gate_id=src_id,
                            to_gate_id=gate_id,
                            to_input_index=idx,
                            length_nm=float(length),
                        )
                    )

        return layout

    def map_half_adder(self) -> ChipLayout:
        """Convenience: decompose and map a half-adder."""
        return self.place_and_route(self.decompose_half_adder(), "half-adder")

    def map_full_adder(self) -> ChipLayout:
        """Convenience: decompose and map a full-adder."""
        return self.place_and_route(self.decompose_full_adder(), "full-adder")
