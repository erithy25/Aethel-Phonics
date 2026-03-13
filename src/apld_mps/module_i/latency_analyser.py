"""Latency Analyser — simulate light-pulse propagation through Transformer layers.

Computes the theoretical end-to-end latency for a single forward pass
through all layers of a neural network mapped onto the Aethel polariton
architecture, and derives the maximum tokens-per-second throughput.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..constants import C_LIGHT, NM_TO_M, PS_TO_S
from .model_parser import ComputeGraph, ComputeNode, OpType
from .polaritonic_mapping import (
    PolaritonicMapper,
    PolaritonicMapping,
    GateCluster,
    ClusterType,
    _GATE_Z_NM,
)
from .memory_interface import HolographicMemoryInterface, SCOCellSpec


@dataclass
class LayerLatency:
    """Latency breakdown for a single Transformer layer.

    All times in picoseconds.

    Attributes:
        layer_index: Transformer layer index.
        attention_ps: Time for the attention sub-block.
        ffn_ps: Time for the feed-forward sub-block.
        memory_read_ps: Time to read weights from holographic memory.
        propagation_ps: Light travel through the layer's z-span.
        total_ps: Sum of all components.
    """

    layer_index: int = 0
    attention_ps: float = 0.0
    ffn_ps: float = 0.0
    memory_read_ps: float = 0.0
    propagation_ps: float = 0.0

    @property
    def total_ps(self) -> float:
        return self.attention_ps + self.ffn_ps + self.memory_read_ps + self.propagation_ps


@dataclass
class LatencyReport:
    """Complete latency analysis for a model on the Aethel chip.

    Attributes:
        model_name: Source model name.
        layer_latencies: Per-layer breakdown.
        embedding_ps: Embedding lookup latency [ps].
        total_latency_ps: End-to-end forward-pass latency [ps].
        total_latency_ns: Same in nanoseconds.
        total_latency_us: Same in microseconds.
        tokens_per_second: Theoretical maximum throughput.
        speed_of_light_in_medium_m_s: Effective light speed in sapphire.
    """

    model_name: str = "unnamed"
    layer_latencies: list[LayerLatency] = field(default_factory=list)
    embedding_ps: float = 0.0
    total_latency_ps: float = 0.0
    total_latency_ns: float = 0.0
    total_latency_us: float = 0.0
    tokens_per_second: float = 0.0
    speed_of_light_in_medium_m_s: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== Latency Analysis: {self.model_name} ===",
            f"Light speed in sapphire:  {self.speed_of_light_in_medium_m_s:.3e} m/s",
            f"Embedding latency:        {self.embedding_ps:.2f} ps",
            f"Transformer layers:       {len(self.layer_latencies)}",
        ]
        for ll in self.layer_latencies[:3]:
            lines.append(
                f"  Layer {ll.layer_index}: attn={ll.attention_ps:.1f} ps  "
                f"ffn={ll.ffn_ps:.1f} ps  mem={ll.memory_read_ps:.1f} ps  "
                f"prop={ll.propagation_ps:.1f} ps  total={ll.total_ps:.1f} ps"
            )
        if len(self.layer_latencies) > 3:
            lines.append(f"  ... ({len(self.layer_latencies) - 3} more layers)")
        lines.extend([
            f"Total latency:            {self.total_latency_ps:.2f} ps "
            f"({self.total_latency_ns:.4f} ns / {self.total_latency_us:.6f} µs)",
            f"Tokens per second:        {self.tokens_per_second:,.0f}",
        ])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate-cluster propagation time estimates [ps]
# ---------------------------------------------------------------------------

# Time for a light pulse to traverse an interference grid (MatMul)
# Depends on the number of z-layers the cluster spans.
_INTERFERENCE_LATENCY_PER_LAYER_PS = 0.05  # 50 fs per layer

# Time for a bistable polariton switch to settle (ReLU/GELU)
_NONLINEAR_SWITCH_PS = 0.5  # 500 fs

# Time for normalisation cascade (Softmax/LayerNorm)
_NORMALISATION_PS = 1.0  # 1 ps ring-resonator feedback loop

# Elementwise coupler (Add/Mul) — essentially speed-of-light
_ELEMENTWISE_PS = 0.02  # 20 fs

# Embedding hologram recall
_EMBEDDING_PS = 0.1  # 100 fs


class LatencyAnalyser:
    """Analyse end-to-end latency of a model on the Aethel chip.

    The analyser traces the critical path through the Transformer
    layer stack, computing the time for each light pulse to:

    1. Read weights from holographic memory.
    2. Propagate through each gate cluster (attention + FFN).
    3. Travel the physical distance between layers in the sapphire.

    Parameters:
        n_sapphire: Refractive index of sapphire at operating wavelength.
        cell_spec: SCO memory cell specification.
        inter_layer_gap_nm: Physical gap between consecutive z-layers [nm].
    """

    def __init__(
        self,
        n_sapphire: float = 1.77,
        cell_spec: SCOCellSpec | None = None,
        inter_layer_gap_nm: float = 5_000.0,
    ) -> None:
        self.n_sapphire = n_sapphire
        self.v_medium = C_LIGHT / n_sapphire  # m/s
        self.cell_spec = cell_spec or SCOCellSpec()
        self.inter_layer_gap_nm = inter_layer_gap_nm

    def _propagation_time_ps(self, distance_nm: float) -> float:
        """Time for light to travel *distance_nm* through sapphire [ps]."""
        dist_m = distance_nm * NM_TO_M
        return (dist_m / self.v_medium) / PS_TO_S

    def _cluster_latency_ps(self, cluster: GateCluster) -> float:
        """Estimate latency for a single gate cluster."""
        if cluster.cluster_type == ClusterType.INTERFERENCE_GRID:
            return cluster.layer_span * _INTERFERENCE_LATENCY_PER_LAYER_PS
        elif cluster.cluster_type == ClusterType.THRESHOLD_NONLINEAR:
            return _NONLINEAR_SWITCH_PS
        elif cluster.cluster_type == ClusterType.NORMALISATION_CASCADE:
            return _NORMALISATION_PS
        elif cluster.cluster_type == ClusterType.ELEMENTWISE_COUPLER:
            return _ELEMENTWISE_PS
        elif cluster.cluster_type == ClusterType.EMBEDDING_HOLOGRAM:
            return _EMBEDDING_PS
        return 0.0  # PASSTHROUGH

    def analyse(
        self,
        graph: ComputeGraph,
        mapping: PolaritonicMapping | None = None,
    ) -> LatencyReport:
        """Run a full latency analysis.

        Parameters:
            graph: Parsed computation graph.
            mapping: Pre-computed polaritonic mapping (computed if None).

        Returns:
            :class:`LatencyReport` with per-layer breakdown and total.
        """
        if mapping is None:
            mapping = PolaritonicMapper().map(graph)

        report = LatencyReport(
            model_name=graph.name,
            speed_of_light_in_medium_m_s=self.v_medium,
        )

        # Build cluster lookup
        cluster_by_node: dict[str, GateCluster] = {
            c.source_node_id: c for c in mapping.clusters
        }

        # Memory read time (one holographic pulse, parallel for all weights)
        mem_read_ps = self.cell_spec.read_time_ps

        # Walk the graph in topological order, grouping by Transformer layer
        nodes = graph.topological_order()

        # Detect layer boundaries by node_id prefix "layerN_"
        current_layer = -1
        attn_ps = 0.0
        ffn_ps = 0.0
        in_ffn = False

        for node in nodes:
            cluster = cluster_by_node.get(node.node_id)
            lat = self._cluster_latency_ps(cluster) if cluster else 0.0

            # Check if this is an embedding node
            if node.op_type == OpType.EMBEDDING:
                report.embedding_ps += lat + mem_read_ps
                continue

            # Parse layer index from node_id
            layer_idx = self._parse_layer_index(node.node_id)

            if layer_idx is not None and layer_idx != current_layer:
                # Save previous layer
                if current_layer >= 0:
                    prop = self._propagation_time_ps(self.inter_layer_gap_nm)
                    ll = LayerLatency(
                        layer_index=current_layer,
                        attention_ps=attn_ps,
                        ffn_ps=ffn_ps,
                        memory_read_ps=mem_read_ps,
                        propagation_ps=prop,
                    )
                    report.layer_latencies.append(ll)
                current_layer = layer_idx
                attn_ps = 0.0
                ffn_ps = 0.0
                in_ffn = False

            # Heuristic: nodes with "ff" or "gelu" in id are FFN
            nid = node.node_id.lower()
            if "ff" in nid or "gelu" in nid:
                in_ffn = True
            elif "ln2" in nid:
                in_ffn = True
            elif "qkv" in nid or "attn" in nid or "softmax" in nid or "ln1" in nid or "o_proj" in nid:
                in_ffn = False

            if in_ffn:
                ffn_ps += lat
            else:
                attn_ps += lat

        # Flush last layer
        if current_layer >= 0:
            prop = self._propagation_time_ps(self.inter_layer_gap_nm)
            ll = LayerLatency(
                layer_index=current_layer,
                attention_ps=attn_ps,
                ffn_ps=ffn_ps,
                memory_read_ps=mem_read_ps,
                propagation_ps=prop,
            )
            report.layer_latencies.append(ll)

        # Total
        report.total_latency_ps = report.embedding_ps + sum(
            ll.total_ps for ll in report.layer_latencies
        )
        report.total_latency_ns = report.total_latency_ps * 1e-3
        report.total_latency_us = report.total_latency_ps * 1e-6

        # Tokens per second
        if report.total_latency_ps > 0:
            latency_s = report.total_latency_ps * PS_TO_S
            report.tokens_per_second = 1.0 / latency_s

        return report

    def analyse_from_spec(
        self,
        *,
        name: str = "model",
        n_layers: int = 32,
        d_model: int = 4096,
        n_heads: int = 32,
        d_ff: int | None = None,
        vocab_size: int = 32000,
        seq_len: int = 2048,
    ) -> LatencyReport:
        """Convenience: build graph from Transformer spec and analyse."""
        from .model_parser import ModelParser

        parser = ModelParser()
        graph = parser.from_transformer_spec(
            name=name,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            vocab_size=vocab_size,
            seq_len=seq_len,
        )
        return self.analyse(graph)

    @staticmethod
    def _parse_layer_index(node_id: str) -> int | None:
        """Extract layer index from node IDs like 'layer5_qkv_42'."""
        if not node_id.startswith("layer"):
            return None
        rest = node_id[5:]  # strip "layer"
        digits = ""
        for ch in rest:
            if ch.isdigit():
                digits += ch
            else:
                break
        return int(digits) if digits else None
