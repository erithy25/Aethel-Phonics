"""Tests for Module D: Logic Compiler and Mapping Layer."""

import pytest

from apld_mps.module_d import GateType, GateLibrary, AutoMapper


class TestGateLibrary:
    def test_all_gates_available(self):
        lib = GateLibrary()
        for gt in GateType:
            gate = lib.get(gt)
            assert gate.gate_type == gt

    def test_xor_truth_table(self):
        lib = GateLibrary()
        xor = lib.get(GateType.XOR)
        outputs = {entry.inputs: entry.output for entry in xor.truth_table}
        assert outputs[(False, False)] is False
        assert outputs[(True, False)] is True
        assert outputs[(False, True)] is True
        assert outputs[(True, True)] is False

    def test_and_truth_table(self):
        lib = GateLibrary()
        gate = lib.get(GateType.AND)
        outputs = {entry.inputs: entry.output for entry in gate.truth_table}
        assert outputs[(True, True)] is True
        assert outputs[(True, False)] is False

    def test_not_truth_table(self):
        lib = GateLibrary()
        gate = lib.get(GateType.NOT)
        outputs = {entry.inputs: entry.output for entry in gate.truth_table}
        assert outputs[(True,)] is False
        assert outputs[(False,)] is True


class TestReversibleGates:
    """Tests for Toffoli (CCNOT) and Fredkin (CSWAP) reversible gates."""

    def test_toffoli_available(self):
        lib = GateLibrary()
        gate = lib.get(GateType.TOFFOLI)
        assert gate.gate_type == GateType.TOFFOLI
        assert gate.reversible is True

    def test_fredkin_available(self):
        lib = GateLibrary()
        gate = lib.get(GateType.FREDKIN)
        assert gate.gate_type == GateType.FREDKIN
        assert gate.reversible is True

    def test_toffoli_truth_table_complete(self):
        lib = GateLibrary()
        gate = lib.get(GateType.TOFFOLI)
        assert len(gate.truth_table) == 8  # 2^3 input combos
        # Verify CCNOT: output_c = c XOR (a AND b)
        for entry in gate.truth_table:
            a, b, c = entry.inputs
            expected_c = c ^ (a and b)
            assert entry.outputs is not None
            assert entry.outputs == (a, b, expected_c)
            assert entry.output == expected_c  # primary output = target

    def test_fredkin_truth_table_complete(self):
        lib = GateLibrary()
        gate = lib.get(GateType.FREDKIN)
        assert len(gate.truth_table) == 8
        # Verify CSWAP: if c then swap(a, b)
        for entry in gate.truth_table:
            c, a, b = entry.inputs
            exp_a = b if c else a
            exp_b = a if c else b
            assert entry.outputs is not None
            assert entry.outputs == (c, exp_a, exp_b)

    def test_toffoli_is_reversible_bijection(self):
        """A reversible gate must be a bijection: every output tuple is unique."""
        lib = GateLibrary()
        gate = lib.get(GateType.TOFFOLI)
        output_set = {entry.outputs for entry in gate.truth_table}
        assert len(output_set) == 8  # all 8 outputs distinct

    def test_fredkin_is_reversible_bijection(self):
        lib = GateLibrary()
        gate = lib.get(GateType.FREDKIN)
        output_set = {entry.outputs for entry in gate.truth_table}
        assert len(output_set) == 8

    def test_toffoli_three_inputs_three_outputs(self):
        lib = GateLibrary()
        gate = lib.get(GateType.TOFFOLI)
        assert gate.n_inputs == 3
        assert gate.n_outputs == 3

    def test_fredkin_three_inputs_three_outputs(self):
        lib = GateLibrary()
        gate = lib.get(GateType.FREDKIN)
        assert gate.n_inputs == 3
        assert gate.n_outputs == 3

    def test_irreversible_gates_not_marked_reversible(self):
        lib = GateLibrary()
        for gt in [GateType.AND, GateType.OR, GateType.NOT, GateType.XOR, GateType.NAND]:
            gate = lib.get(gt)
            assert gate.reversible is False

    def test_all_gates_still_available(self):
        """Ensure adding reversible gates didn't break existing gates."""
        lib = GateLibrary()
        assert len(lib.list_gates()) == 7  # 5 classic + 2 reversible
        for gt in GateType:
            gate = lib.get(gt)
            assert gate.gate_type == gt


class TestAutoMapper:
    def test_half_adder_layout(self):
        mapper = AutoMapper()
        layout = mapper.map_half_adder()
        assert layout.gate_count == 2
        assert layout.name == "half-adder"

    def test_full_adder_layout(self):
        mapper = AutoMapper()
        layout = mapper.map_full_adder()
        assert layout.gate_count == 5
        assert layout.name == "full-adder"
        assert len(layout.wires) > 0

    def test_wire_lengths_positive(self):
        mapper = AutoMapper()
        layout = mapper.map_full_adder()
        for wire in layout.wires:
            assert wire.length_nm >= 0
