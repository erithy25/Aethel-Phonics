"""Detail 6.2 — Petabit-scale I/O dashboard.

Real-time monitoring of data streams entering and leaving the Aethel V1,
bottleneck analysis between memory read speed and external network bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class ChannelStats:
    """Statistics for a single I/O channel.

    Attributes:
        name: Channel identifier.
        data_rate_Gbps: Current throughput [Gbit/s].
        capacity_Gbps: Maximum capacity [Gbit/s].
        latency_ns: One-way latency [ns].
        error_rate: Bit-error rate.
    """

    name: str
    data_rate_Gbps: float
    capacity_Gbps: float
    latency_ns: float = 1.0
    error_rate: float = 0.0

    @property
    def utilisation(self) -> float:
        """Channel utilisation [0–1]."""
        if self.capacity_Gbps <= 0:
            return 0.0
        return min(self.data_rate_Gbps / self.capacity_Gbps, 1.0)


@dataclass
class BottleneckReport:
    """I/O bottleneck analysis result.

    Attributes:
        bottleneck_channel: Name of the limiting channel.
        bottleneck_rate_Gbps: Throughput of the limiting channel [Gbit/s].
        aggregate_input_Gbps: Total ingress bandwidth [Gbit/s].
        aggregate_output_Gbps: Total egress bandwidth [Gbit/s].
        memory_read_rate_Gbps: Internal memory read rate [Gbit/s].
        network_bandwidth_Gbps: External network bandwidth [Gbit/s].
        is_memory_bound: True if memory read is the bottleneck.
        is_network_bound: True if external network is the bottleneck.
        headroom_fraction: Remaining capacity as fraction of bottleneck.
    """

    bottleneck_channel: str
    bottleneck_rate_Gbps: float
    aggregate_input_Gbps: float
    aggregate_output_Gbps: float
    memory_read_rate_Gbps: float
    network_bandwidth_Gbps: float
    is_memory_bound: bool
    is_network_bound: bool
    headroom_fraction: float


class PetabitIODashboard:
    """Real-time I/O monitoring and bottleneck detection.

    Tracks multiple parallel fibre channels and internal memory buses,
    identifies throughput bottlenecks, and computes aggregate bandwidth.
    """

    def __init__(
        self,
        memory_read_rate_Gbps: float = 100_000.0,  # 100 Tbit/s internal
        network_bandwidth_Gbps: float = 10_000.0,  # 10 Tbit/s external
    ) -> None:
        self.memory_rate = memory_read_rate_Gbps
        self.network_bw = network_bandwidth_Gbps
        self._input_channels: list[ChannelStats] = []
        self._output_channels: list[ChannelStats] = []
        self._history: list[dict[str, float]] = []

    def add_input_channel(self, channel: ChannelStats) -> None:
        self._input_channels.append(channel)

    def add_output_channel(self, channel: ChannelStats) -> None:
        self._output_channels.append(channel)

    @property
    def aggregate_input_Gbps(self) -> float:
        return sum(ch.data_rate_Gbps for ch in self._input_channels)

    @property
    def aggregate_output_Gbps(self) -> float:
        return sum(ch.data_rate_Gbps for ch in self._output_channels)

    @property
    def aggregate_input_Pbps(self) -> float:
        return self.aggregate_input_Gbps / 1e6

    @property
    def aggregate_output_Pbps(self) -> float:
        return self.aggregate_output_Gbps / 1e6

    def record_snapshot(self) -> None:
        """Record current bandwidth snapshot for time-series analysis."""
        self._history.append({
            "input_Gbps": self.aggregate_input_Gbps,
            "output_Gbps": self.aggregate_output_Gbps,
        })

    def find_bottleneck(self) -> BottleneckReport:
        """Identify the throughput bottleneck.

        Compares:
        - Aggregate I/O throughput
        - Internal memory read rate
        - External network bandwidth
        """
        agg_in = self.aggregate_input_Gbps
        agg_out = self.aggregate_output_Gbps

        # Find the limiting factor
        rates = {
            "memory_read": self.memory_rate,
            "network_egress": self.network_bw,
            "input_aggregate": sum(ch.capacity_Gbps for ch in self._input_channels) or float("inf"),
            "output_aggregate": sum(ch.capacity_Gbps for ch in self._output_channels) or float("inf"),
        }

        bottleneck_name = min(rates, key=rates.get)  # type: ignore[arg-type]
        bottleneck_rate = rates[bottleneck_name]

        is_mem = bottleneck_name == "memory_read"
        is_net = bottleneck_name == "network_egress"

        current_demand = max(agg_in, agg_out)
        headroom = 1.0 - current_demand / bottleneck_rate if bottleneck_rate > 0 else 0.0

        return BottleneckReport(
            bottleneck_channel=bottleneck_name,
            bottleneck_rate_Gbps=bottleneck_rate,
            aggregate_input_Gbps=agg_in,
            aggregate_output_Gbps=agg_out,
            memory_read_rate_Gbps=self.memory_rate,
            network_bandwidth_Gbps=self.network_bw,
            is_memory_bound=is_mem,
            is_network_bound=is_net,
            headroom_fraction=max(0.0, headroom),
        )

    def channel_summary(self) -> str:
        """Human-readable summary of all channels."""
        lines = [f"{'Channel':<20} {'Rate (Gbps)':>12} {'Cap (Gbps)':>12} {'Util':>8}"]
        lines.append("-" * 56)
        for ch in self._input_channels + self._output_channels:
            lines.append(
                f"{ch.name:<20} {ch.data_rate_Gbps:>12.1f} {ch.capacity_Gbps:>12.1f} "
                f"{ch.utilisation * 100:>6.1f}%"
            )
        lines.append(f"\nAggregate Input:  {self.aggregate_input_Gbps:.1f} Gbps")
        lines.append(f"Aggregate Output: {self.aggregate_output_Gbps:.1f} Gbps")
        return "\n".join(lines)
