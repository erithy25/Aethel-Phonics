"""Tests for Module A: Material & Geometry Editor."""

import numpy as np
import pytest

from apld_mps.module_a import (
    SubstrateDatabase,
    TMDMonolayer,
    ExcitonProperties,
    WaveguideGeometry,
    Channel,
    YSplitter,
    RingResonator,
    InteractionZone,
)
from apld_mps.module_a.tmd_layer import WSe2, MoSe2
from apld_mps.module_a.geometry import Point


class TestSubstrateDatabase:
    def test_sapphire_registered(self):
        db = SubstrateDatabase()
        assert "Sapphire (Al₂O₃)" in db.list_materials()

    def test_sapphire_refractive_index_reasonable(self):
        db = SubstrateDatabase()
        sapphire = db.get("Sapphire (Al₂O₃)")
        n = sapphire.n(700.0)  # 700 nm — visible red
        assert 1.7 < n < 1.8  # sapphire ordinary ray ~1.76

    def test_sapphire_n_dispersion(self):
        db = SubstrateDatabase()
        sapphire = db.get("Sapphire (Al₂O₃)")
        n_blue = sapphire.n(450.0)
        n_red = sapphire.n(700.0)
        assert n_blue > n_red  # normal dispersion

    def test_missing_substrate_raises(self):
        db = SubstrateDatabase()
        with pytest.raises(KeyError):
            db.get("Unobtanium")


class TestTMDMonolayer:
    def test_wse2_defaults(self):
        assert WSe2.name == "WSe2"
        assert WSe2.exciton.resonance_energy_eV == pytest.approx(1.72)
        assert WSe2.total_linewidth > 0

    def test_mose2_defaults(self):
        assert MoSe2.name == "MoSe2"
        assert MoSe2.exciton.resonance_energy_eV == pytest.approx(1.66)

    def test_resonance_freq_positive(self):
        assert WSe2.exciton.resonance_freq_rad > 0


class TestGeometry:
    def test_channel_length(self):
        ch = Channel(points=[Point(0, 0), Point(1000, 0), Point(1000, 500)])
        assert ch.length_nm == pytest.approx(1500.0)

    def test_ring_circumference(self):
        ring = RingResonator(centre=Point(0, 0), radius_nm=1000)
        assert ring.circumference_nm == pytest.approx(2 * np.pi * 1000)

    def test_xor_interferometer(self):
        geo = WaveguideGeometry.xor_interferometer()
        assert "XOR" in geo.name
        assert len(geo.components) == 3  # 2 Y-splitters + 1 interaction zone

    def test_bounding_box(self):
        geo = WaveguideGeometry()
        geo.add(Channel(points=[Point(0, 0), Point(100, 200)]))
        mn, mx = geo.bounding_box
        assert mn.x == 0 and mn.y == 0
        assert mx.x == 100 and mx.y == 200
