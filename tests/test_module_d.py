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
