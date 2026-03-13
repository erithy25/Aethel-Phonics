"""Tests for Module C upgrades: fibre interface and petabit I/O dashboard."""

import numpy as np
import pytest

from apld_mps.module_c.fiber_interface import (
    HollowCoreFibreInterface,
    FibreSpec,
    CouplingResult,
)
from apld_mps.module_c.io_dashboard import (
    PetabitIODashboard,
    ChannelStats,
    BottleneckReport,
)


class TestHollowCoreFibreInterface:
    def test_fresnel_reflection(self):
        iface = HollowCoreFibreInterface(FibreSpec())
        R = iface.fresnel_reflection(1.0, 1.76)
        assert 0 < R < 0.1  # air→sapphire ~ 7.7%

    def test_ar_coating_reduces_reflection(self):
        iface = HollowCoreFibreInterface(FibreSpec())
        R_bare = iface.fresnel_reflection(1.0, 1.76)
        R_ar = iface.ar_coated_reflection()
        assert R_ar < R_bare

    def test_mode_overlap_positive(self):
        iface = HollowCoreFibreInterface(FibreSpec())
        overlap = iface.mode_overlap_integral()
        assert 0 < overlap <= 1

    def test_taper_length_positive(self):
        iface = HollowCoreFibreInterface(FibreSpec())
        length = iface.optimal_taper_length_um()
        assert length > 0

    def test_analyse_coupling(self):
        iface = HollowCoreFibreInterface(FibreSpec())
        result = iface.analyse_coupling()
        assert isinstance(result, CouplingResult)
        assert 0 < result.coupling_efficiency <= 1
        assert result.insertion_loss_dB > 0
        assert result.reflection_loss >= 0

    def test_ar_target_below_01_percent(self):
        """AR coating should get reflection well below 1%."""
        iface = HollowCoreFibreInterface(FibreSpec())
        R = iface.ar_coated_reflection()
        assert R < 0.01  # < 1%


class TestPetabitIODashboard:
    def test_add_channels(self):
        dash = PetabitIODashboard()
        dash.add_input_channel(ChannelStats("fibre-1", 100.0, 400.0))
        dash.add_output_channel(ChannelStats("fibre-2", 200.0, 400.0))
        assert dash.aggregate_input_Gbps == pytest.approx(100.0)
        assert dash.aggregate_output_Gbps == pytest.approx(200.0)

    def test_utilisation(self):
        ch = ChannelStats("test", 200.0, 400.0)
        assert ch.utilisation == pytest.approx(0.5)

    def test_bottleneck_detection(self):
        dash = PetabitIODashboard(
            memory_read_rate_Gbps=1000.0,
            network_bandwidth_Gbps=500.0,
        )
        dash.add_input_channel(ChannelStats("in-1", 100.0, 200.0))
        dash.add_output_channel(ChannelStats("out-1", 100.0, 200.0))
        report = dash.find_bottleneck()
        assert isinstance(report, BottleneckReport)
        assert report.bottleneck_rate_Gbps > 0
        assert report.headroom_fraction >= 0

    def test_network_bound(self):
        dash = PetabitIODashboard(
            memory_read_rate_Gbps=100_000.0,
            network_bandwidth_Gbps=100.0,  # very limited network
        )
        dash.add_input_channel(ChannelStats("in", 50.0, 1000.0))
        report = dash.find_bottleneck()
        assert report.is_network_bound

    def test_snapshot_recording(self):
        dash = PetabitIODashboard()
        dash.add_input_channel(ChannelStats("in", 100.0, 400.0))
        dash.record_snapshot()
        dash.record_snapshot()
        assert len(dash._history) == 2

    def test_channel_summary(self):
        dash = PetabitIODashboard()
        dash.add_input_channel(ChannelStats("fibre-in", 100.0, 400.0))
        summary = dash.channel_summary()
        assert "fibre-in" in summary
        assert "100.0" in summary
