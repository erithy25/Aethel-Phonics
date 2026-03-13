"""Sektion 4 — Petabit I/O & Fibre Dashboard panel.

Real-time throughput meters, bottleneck radar, and fibre coupling
efficiency status, integrating Module C's I/O dashboard and fibre
interface models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from ..module_c.io_dashboard import PetabitIODashboard, ChannelStats, BottleneckReport
from ..module_c.fiber_interface import (
    HollowCoreFibreInterface,
    FibreSpec,
    CouplingResult,
)


class BottleneckType(Enum):
    """Visual indicator for bottleneck radar."""
    NONE = auto()
    MEMORY_BOUND = auto()
    NETWORK_BOUND = auto()
    INPUT_BOUND = auto()
    OUTPUT_BOUND = auto()


@dataclass
class ThroughputMeterData:
    """Data for the real-time throughput meters.

    Attributes:
        input_Gbps: Current aggregate input [Gbit/s].
        output_Gbps: Current aggregate output [Gbit/s].
        input_Pbps: Input in petabit/s for display.
        output_Pbps: Output in petabit/s for display.
        input_channels: Per-channel stats.
        output_channels: Per-channel stats.
        history: Time-series of (input, output) snapshots.
    """

    input_Gbps: float
    output_Gbps: float
    input_Pbps: float
    output_Pbps: float
    input_channels: list[dict]
    output_channels: list[dict]
    history: list[dict[str, float]]


@dataclass
class BottleneckRadarData:
    """Bottleneck radar visualisation data.

    Attributes:
        bottleneck_type: Which subsystem is limiting throughput.
        bottleneck_name: Human-readable bottleneck label.
        bottleneck_rate_Gbps: Rate of the limiting channel [Gbit/s].
        headroom_percent: Remaining capacity [%].
        memory_rate_Gbps: Internal memory read speed [Gbit/s].
        network_rate_Gbps: External network bandwidth [Gbit/s].
    """

    bottleneck_type: BottleneckType
    bottleneck_name: str
    bottleneck_rate_Gbps: float
    headroom_percent: float
    memory_rate_Gbps: float
    network_rate_Gbps: float


@dataclass
class FibreCouplingStatus:
    """Status of all fibre-to-sapphire coupling interfaces.

    Attributes:
        interfaces: List of per-interface coupling results.
        worst_insertion_loss_dB: Highest IL across all interfaces.
        best_coupling_efficiency: Best η across all interfaces.
    """

    interfaces: list[dict]
    worst_insertion_loss_dB: float
    best_coupling_efficiency: float


class IOPanel:
    """Petabit I/O & fibre dashboard panel controller.

    Manages throughput monitoring, bottleneck detection, and fibre
    coupling efficiency for all I/O ports.
    """

    def __init__(
        self,
        memory_read_rate_Gbps: float = 100_000.0,
        network_bandwidth_Gbps: float = 10_000.0,
    ) -> None:
        self._dashboard = PetabitIODashboard(
            memory_read_rate_Gbps=memory_read_rate_Gbps,
            network_bandwidth_Gbps=network_bandwidth_Gbps,
        )
        self._fibre_interfaces: list[tuple[str, HollowCoreFibreInterface]] = []
        self._coupling_results: dict[str, CouplingResult] = {}

    # --- Channel management ---

    def add_input_channel(
        self, name: str, data_rate_Gbps: float, capacity_Gbps: float,
        latency_ns: float = 1.0,
    ) -> None:
        self._dashboard.add_input_channel(
            ChannelStats(name, data_rate_Gbps, capacity_Gbps, latency_ns)
        )

    def add_output_channel(
        self, name: str, data_rate_Gbps: float, capacity_Gbps: float,
        latency_ns: float = 1.0,
    ) -> None:
        self._dashboard.add_output_channel(
            ChannelStats(name, data_rate_Gbps, capacity_Gbps, latency_ns)
        )

    # --- Fibre interface management ---

    def add_fibre_interface(
        self,
        name: str,
        fibre: FibreSpec | None = None,
        sapphire_n: float = 1.76,
        waveguide_width_nm: float = 200.0,
    ) -> CouplingResult:
        """Add and analyse a fibre-to-sapphire interface."""
        spec = fibre or FibreSpec()
        interface = HollowCoreFibreInterface(spec, sapphire_n, waveguide_width_nm)
        self._fibre_interfaces.append((name, interface))
        result = interface.analyse_coupling()
        self._coupling_results[name] = result
        return result

    # --- Throughput meter ---

    def record_snapshot(self) -> None:
        """Record current bandwidth snapshot."""
        self._dashboard.record_snapshot()

    def get_throughput_data(self) -> ThroughputMeterData:
        """Produce throughput meter display data."""
        in_channels = []
        for ch in self._dashboard._input_channels:
            in_channels.append({
                "name": ch.name,
                "rate_Gbps": ch.data_rate_Gbps,
                "capacity_Gbps": ch.capacity_Gbps,
                "utilisation": ch.utilisation,
            })
        out_channels = []
        for ch in self._dashboard._output_channels:
            out_channels.append({
                "name": ch.name,
                "rate_Gbps": ch.data_rate_Gbps,
                "capacity_Gbps": ch.capacity_Gbps,
                "utilisation": ch.utilisation,
            })

        return ThroughputMeterData(
            input_Gbps=self._dashboard.aggregate_input_Gbps,
            output_Gbps=self._dashboard.aggregate_output_Gbps,
            input_Pbps=self._dashboard.aggregate_input_Pbps,
            output_Pbps=self._dashboard.aggregate_output_Pbps,
            input_channels=in_channels,
            output_channels=out_channels,
            history=list(self._dashboard._history),
        )

    # --- Bottleneck radar ---

    def get_bottleneck_radar(self) -> BottleneckRadarData:
        """Analyse and return bottleneck radar data."""
        report = self._dashboard.find_bottleneck()

        if report.is_memory_bound:
            bt = BottleneckType.MEMORY_BOUND
        elif report.is_network_bound:
            bt = BottleneckType.NETWORK_BOUND
        elif "input" in report.bottleneck_channel:
            bt = BottleneckType.INPUT_BOUND
        elif "output" in report.bottleneck_channel:
            bt = BottleneckType.OUTPUT_BOUND
        else:
            bt = BottleneckType.NONE

        return BottleneckRadarData(
            bottleneck_type=bt,
            bottleneck_name=report.bottleneck_channel,
            bottleneck_rate_Gbps=report.bottleneck_rate_Gbps,
            headroom_percent=report.headroom_fraction * 100,
            memory_rate_Gbps=report.memory_read_rate_Gbps,
            network_rate_Gbps=report.network_bandwidth_Gbps,
        )

    # --- Fibre coupling status ---

    def get_coupling_status(self) -> FibreCouplingStatus:
        """Produce fibre coupling efficiency status display."""
        interfaces = []
        worst_il = 0.0
        best_eff = 0.0

        for name, _ in self._fibre_interfaces:
            result = self._coupling_results.get(name)
            if result:
                interfaces.append({
                    "name": name,
                    "coupling_efficiency": result.coupling_efficiency,
                    "reflection_loss": result.reflection_loss,
                    "insertion_loss_dB": result.insertion_loss_dB,
                    "taper_length_um": result.optimised_taper_length_um,
                })
                worst_il = max(worst_il, result.insertion_loss_dB)
                best_eff = max(best_eff, result.coupling_efficiency)

        return FibreCouplingStatus(
            interfaces=interfaces,
            worst_insertion_loss_dB=worst_il,
            best_coupling_efficiency=best_eff,
        )
