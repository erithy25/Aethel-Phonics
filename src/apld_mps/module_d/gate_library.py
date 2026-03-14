"""Detail 5.1 — Logic gate library.

Each standard gate (AND, OR, NOT, XOR, NAND) is linked to a specific
2D waveguide geometry and a laser-pulse sequence that realises its truth table
via polariton interference and nonlinear switching.

Reversible gates (TOFFOLI, FREDKIN) are additionally provided.  These gates
preserve all input information — no bits are erased — and therefore incur
*zero* Landauer entropy cost (k_B T ln 2 per erased bit).  In a polariton
architecture the practical benefit is the complete elimination of the
thermodynamic heat floor for every operation implemented reversibly.
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
    TOFFOLI = auto()   # CCNOT — reversible, 3-in/3-out
    FREDKIN = auto()   # CSWAP — reversible, 3-in/3-out


@dataclass
class TruthTableEntry:
    """One row of a gate's truth table.

    Attributes:
        inputs: Tuple of boolean input values.
        output: Expected boolean output (first / only output for classic gates).
        pulse_config: Laser pulse configuration that realises this row.
        outputs: Full tuple of output values for multi-output (reversible) gates.
                 When *None* the single ``output`` field is authoritative.
    """

    inputs: tuple[bool, ...]
    output: bool
    pulse_config: LaserSource
    outputs: tuple[bool, ...] | None = None


@dataclass
class LogicGate:
    """A polariton logic gate: geometry + truth-table with pulse sequences.

    Attributes:
        gate_type: AND / OR / NOT / XOR / NAND / TOFFOLI / FREDKIN.
        geometry: The waveguide structure implementing this gate.
        truth_table: Full truth table with corresponding pulse configurations.
        input_positions_nm: Spatial positions of input ports [nm].
        output_position_nm: Spatial position of the (first / only) output port [nm].
        footprint_nm: (width, height) bounding box on the chip [nm].
        output_positions_nm: Positions of *all* output ports for multi-output gates.
                             When *None* the single ``output_position_nm`` is used.
        reversible: Whether this gate preserves all input information (zero
                    Landauer dissipation).
    """

    gate_type: GateType
    geometry: WaveguideGeometry
    truth_table: list[TruthTableEntry]
    input_positions_nm: list[tuple[float, float]]
    output_position_nm: tuple[float, float]
    footprint_nm: tuple[float, float] = (10_000.0, 5_000.0)
    output_positions_nm: list[tuple[float, float]] | None = None
    reversible: bool = False

    @property
    def n_inputs(self) -> int:
        return len(self.input_positions_nm)

    @property
    def n_outputs(self) -> int:
        if self.output_positions_nm is not None:
            return len(self.output_positions_nm)
        return 1


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

        # --- Toffoli gate (CCNOT) — reversible 3-in / 3-out ---
        # Inputs: a (control-1), b (control-2), c (target)
        # Outputs: a, b, c XOR (a AND b)
        # No bits are erased → zero Landauer dissipation.
        #
        # Polariton implementation: two-stage nonlinear cascade.
        # Stage 1: a AND b via nonlinear threshold in an interaction zone.
        # Stage 2: result XOR c via balanced MZI interferometer.
        # Both control inputs are pass-through (waveguide copy).
        tof_in_a = (0.0, 2000.0)
        tof_in_b = (0.0, 0.0)
        tof_in_c = (0.0, -2000.0)
        tof_out_a = (20000.0, 2000.0)
        tof_out_b = (20000.0, 0.0)
        tof_out_c = (20000.0, -2000.0)

        tof_geo = WaveguideGeometry(name="TOFFOLI-reversible-ccnot")
        # Stage 1: nonlinear AND of a,b
        tof_geo.add(InteractionZone(
            centre=Point(6000, 1000), length_nm=2000, separation_nm=100,
        ))
        # Stage 2: XOR interference of AND-result with c
        tof_geo.add(InteractionZone(
            centre=Point(14000, -1000), length_nm=2000, separation_nm=100,
        ))
        # Pass-through channels for control outputs
        tof_geo.add(Channel(points=[Point(0, 2000), Point(20000, 2000)]))
        tof_geo.add(Channel(points=[Point(0, 0), Point(20000, 0)]))

        tof_tt = []
        for a_on, b_on, c_on in [
            (False, False, False), (False, False, True),
            (False, True, False),  (False, True, True),
            (True, False, False),  (True, False, True),
            (True, True, False),   (True, True, True),
        ]:
            src = LaserSource()
            src.add_pulse(_make_pulse(e, tof_in_a, a_on))
            src.add_pulse(_make_pulse(e, tof_in_b, b_on))
            src.add_pulse(_make_pulse(e, tof_in_c, c_on))
            out_c = c_on ^ (a_on and b_on)
            tof_tt.append(TruthTableEntry(
                inputs=(a_on, b_on, c_on),
                output=out_c,  # primary output = target bit
                pulse_config=src,
                outputs=(a_on, b_on, out_c),
            ))
        self._gates[GateType.TOFFOLI] = LogicGate(
            gate_type=GateType.TOFFOLI,
            geometry=tof_geo,
            truth_table=tof_tt,
            input_positions_nm=[tof_in_a, tof_in_b, tof_in_c],
            output_position_nm=tof_out_c,
            footprint_nm=(20000.0, 6000.0),
            output_positions_nm=[tof_out_a, tof_out_b, tof_out_c],
            reversible=True,
        )

        # --- Fredkin gate (CSWAP) — reversible 3-in / 3-out ---
        # Inputs: c (control), a (target-1), b (target-2)
        # Outputs: c, (c ? b : a), (c ? a : b)
        # When c=1 the two targets are swapped; when c=0 they pass through.
        # No bits erased → zero Landauer dissipation.
        #
        # Polariton implementation: control-dependent directional coupler.
        # A nonlinear-threshold stage reads c.  Its output modulates the
        # coupling coefficient of a balanced directional coupler that
        # either swaps or passes through the a/b polariton packets.
        fred_in_c = (0.0, 2000.0)
        fred_in_a = (0.0, 0.0)
        fred_in_b = (0.0, -2000.0)
        fred_out_c = (20000.0, 2000.0)
        fred_out_a = (20000.0, 0.0)
        fred_out_b = (20000.0, -2000.0)

        fred_geo = WaveguideGeometry(name="FREDKIN-reversible-cswap")
        # Control readout via nonlinear threshold
        fred_geo.add(InteractionZone(
            centre=Point(6000, 2000), length_nm=2000, separation_nm=100,
        ))
        # Directional coupler for conditional swap
        fred_geo.add(InteractionZone(
            centre=Point(14000, -1000), length_nm=4000, separation_nm=80,
        ))
        # Control pass-through
        fred_geo.add(Channel(points=[Point(0, 2000), Point(20000, 2000)]))

        fred_tt = []
        for c_on, a_on, b_on in [
            (False, False, False), (False, False, True),
            (False, True, False),  (False, True, True),
            (True, False, False),  (True, False, True),
            (True, True, False),   (True, True, True),
        ]:
            src = LaserSource()
            src.add_pulse(_make_pulse(e, fred_in_c, c_on))
            src.add_pulse(_make_pulse(e, fred_in_a, a_on))
            src.add_pulse(_make_pulse(e, fred_in_b, b_on))
            out_a = b_on if c_on else a_on
            out_b = a_on if c_on else b_on
            fred_tt.append(TruthTableEntry(
                inputs=(c_on, a_on, b_on),
                output=out_a,  # primary output = first target
                pulse_config=src,
                outputs=(c_on, out_a, out_b),
            ))
        self._gates[GateType.FREDKIN] = LogicGate(
            gate_type=GateType.FREDKIN,
            geometry=fred_geo,
            truth_table=fred_tt,
            input_positions_nm=[fred_in_c, fred_in_a, fred_in_b],
            output_position_nm=fred_out_a,
            footprint_nm=(20000.0, 6000.0),
            output_positions_nm=[fred_out_c, fred_out_a, fred_out_b],
            reversible=True,
        )

    def get(self, gate_type: GateType) -> LogicGate:
        return self._gates[gate_type]

    def list_gates(self) -> list[GateType]:
        return list(self._gates.keys())
