"""Detail 5.1 — Logic gate library.

Each standard gate (AND, OR, NOT, XOR, NAND) is linked to a specific
2D waveguide geometry and a laser-pulse sequence that realises its truth table
via polariton interference and nonlinear switching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

from ..module_a.geometry import WaveguideGeometry, Point, Channel, YSplitter, InteractionZone
from ..module_c.laser_source import LaserSource, LaserPulse


class GateType(Enum):
    AND = auto()
    OR = auto()
    NOT = auto()
    XOR = auto()
    NAND = auto()


@dataclass
class TruthTableEntry:
    """One row of a gate's truth table.

    Attributes:
        inputs: Tuple of boolean input values.
        output: Expected boolean output.
        pulse_config: Laser pulse configuration that realises this row.
    """

    inputs: tuple[bool, ...]
    output: bool
    pulse_config: LaserSource


@dataclass
class LogicGate:
    """A polariton logic gate: geometry + truth-table with pulse sequences.

    Attributes:
        gate_type: AND / OR / NOT / XOR / NAND.
        geometry: The waveguide structure implementing this gate.
        truth_table: Full truth table with corresponding pulse configurations.
        input_positions_nm: Spatial positions of input ports [nm].
        output_position_nm: Spatial position of output port [nm].
        footprint_nm: (width, height) bounding box on the chip [nm].
    """

    gate_type: GateType
    geometry: WaveguideGeometry
    truth_table: list[TruthTableEntry]
    input_positions_nm: list[tuple[float, float]]
    output_position_nm: tuple[float, float]
    footprint_nm: tuple[float, float] = (10_000.0, 5_000.0)

    @property
    def n_inputs(self) -> int:
        return len(self.input_positions_nm)


def _make_pulse(energy_eV: float, pos: tuple[float, float], on: bool) -> LaserPulse:
    return LaserPulse(
        energy_eV=energy_eV,
        intensity_W_cm2=1e4 if on else 0.0,
        position_nm=pos,
    )


class GateLibrary:
    """Registry of standard polariton logic gates.

    Pre-populated with AND, OR, NOT, XOR, NAND gates. Each gate carries a
    default waveguide geometry and pulse-sequence truth table.
    """

    def __init__(self, exciton_energy_eV: float = 1.72) -> None:
        self._gates: dict[GateType, LogicGate] = {}
        self._energy = exciton_energy_eV
        self._build_defaults()

    def _build_defaults(self) -> None:
        e = self._energy

        # --- XOR gate (interference-based) ---
        in_a = (0.0, 1000.0)
        in_b = (0.0, -1000.0)
        out = (12000.0, 0.0)

        xor_geo = WaveguideGeometry.xor_interferometer()
        xor_tt = []
        for a_on, b_on in [(False, False), (True, False), (False, True), (True, True)]:
            src = LaserSource()
            src.add_pulse(_make_pulse(e, in_a, a_on))
            src.add_pulse(_make_pulse(e, in_b, b_on))
            xor_tt.append(TruthTableEntry(
                inputs=(a_on, b_on),
                output=a_on ^ b_on,
                pulse_config=src,
            ))
        self._gates[GateType.XOR] = LogicGate(
            gate_type=GateType.XOR,
            geometry=xor_geo,
            truth_table=xor_tt,
            input_positions_nm=[in_a, in_b],
            output_position_nm=out,
        )

        # --- AND gate (nonlinear threshold) ---
        and_geo = WaveguideGeometry(name="AND-nonlinear-threshold")
        and_geo.add(InteractionZone(centre=Point(5000, 0), length_nm=2000, separation_nm=100))
        and_tt = []
        for a_on, b_on in [(False, False), (True, False), (False, True), (True, True)]:
            src = LaserSource()
            src.add_pulse(_make_pulse(e, in_a, a_on))
            src.add_pulse(_make_pulse(e, in_b, b_on))
            and_tt.append(TruthTableEntry(
                inputs=(a_on, b_on),
                output=a_on and b_on,
                pulse_config=src,
            ))
        self._gates[GateType.AND] = LogicGate(
            gate_type=GateType.AND,
            geometry=and_geo,
            truth_table=and_tt,
            input_positions_nm=[in_a, in_b],
            output_position_nm=out,
        )

        # --- OR gate (additive combination) ---
        or_geo = WaveguideGeometry(name="OR-additive")
        or_geo.add(YSplitter(origin=Point(0, 0), angle_deg=60))
        or_tt = []
        for a_on, b_on in [(False, False), (True, False), (False, True), (True, True)]:
            src = LaserSource()
            src.add_pulse(_make_pulse(e, in_a, a_on))
            src.add_pulse(_make_pulse(e, in_b, b_on))
            or_tt.append(TruthTableEntry(
                inputs=(a_on, b_on),
                output=a_on or b_on,
                pulse_config=src,
            ))
        self._gates[GateType.OR] = LogicGate(
            gate_type=GateType.OR,
            geometry=or_geo,
            truth_table=or_tt,
            input_positions_nm=[in_a, in_b],
            output_position_nm=out,
        )

        # --- NOT gate (bistable inverter) ---
        not_in = (0.0, 0.0)
        not_out = (8000.0, 0.0)
        not_geo = WaveguideGeometry(name="NOT-bistable-inverter")
        not_geo.add(Channel(points=[Point(0, 0), Point(8000, 0)]))
        not_tt = []
        for a_on in [False, True]:
            src = LaserSource()
            src.add_pulse(_make_pulse(e, not_in, a_on))
            not_tt.append(TruthTableEntry(
                inputs=(a_on,),
                output=not a_on,
                pulse_config=src,
            ))
        self._gates[GateType.NOT] = LogicGate(
            gate_type=GateType.NOT,
            geometry=not_geo,
            truth_table=not_tt,
            input_positions_nm=[not_in],
            output_position_nm=not_out,
            footprint_nm=(8000.0, 2000.0),
        )

        # --- NAND gate (AND + NOT) ---
        nand_geo = WaveguideGeometry(name="NAND-composite")
        nand_geo.add(InteractionZone(centre=Point(5000, 0), length_nm=2000))
        nand_geo.add(Channel(points=[Point(7000, 0), Point(15000, 0)]))
        nand_tt = []
        for a_on, b_on in [(False, False), (True, False), (False, True), (True, True)]:
            src = LaserSource()
            src.add_pulse(_make_pulse(e, in_a, a_on))
            src.add_pulse(_make_pulse(e, in_b, b_on))
            nand_tt.append(TruthTableEntry(
                inputs=(a_on, b_on),
                output=not (a_on and b_on),
                pulse_config=src,
            ))
        self._gates[GateType.NAND] = LogicGate(
            gate_type=GateType.NAND,
            geometry=nand_geo,
            truth_table=nand_tt,
            input_positions_nm=[in_a, in_b],
            output_position_nm=(15000.0, 0.0),
            footprint_nm=(15000.0, 5000.0),
        )

    def get(self, gate_type: GateType) -> LogicGate:
        return self._gates[gate_type]

    def list_gates(self) -> list[GateType]:
        return list(self._gates.keys())
