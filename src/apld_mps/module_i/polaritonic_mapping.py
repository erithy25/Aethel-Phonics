"""Polaritonic Mapping — translate tensor operations to gate clusters.

Maps each neural-network operation (MatMul, ReLU, Softmax, etc.) onto
specialised polariton-gate clusters that exploit wave-interference patterns
in the sapphire lattice for massively parallel computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .model_parser import ComputeGraph, ComputeNode, OpType


class ClusterType(Enum):
    """Physical gate-cluster archetype in the polariton fabric."""

    INTERFERENCE_GRID = auto()       # Parallel wave-interference for MatMul
    THRESHOLD_NONLINEAR = auto()     # Bistable polariton switch for ReLU/GELU
    NORMALISATION_CASCADE = auto()   # Coupled-resonator chain for Softmax/LayerNorm
    ELEMENTWISE_COUPLER = auto()     # Evanescent coupler for Add/Mul
    EMBEDDING_HOLOGRAM = auto()      # Holographic lookup from spin-crossover memory
    PASSTHROUGH = auto()             # Reshape/Transpose (routing only, no gates)


@dataclass
class GateCluster:
    """A cluster of polariton gates that implements one tensor operation.

    Attributes:
        cluster_id: Unique identifier.
        cluster_type: Physical archetype.
        source_node_id: ID of the :class:`ComputeNode` this cluster implements.
        gate_count: Number of elementary polariton gates required.
        footprint_nm: (x, y, z) bounding box of the cluster [nm].
        layer_span: Number of z-layers this cluster occupies.
        interference_channels: Number of parallel waveguide channels used
            for wave-interference computation (relevant for MatMul).
        power_mw: Estimated optical pump power [mW].
    """

    cluster_id: str
    cluster_type: ClusterType
    source_node_id: str
    gate_count: int = 0
    footprint_nm: tuple[float, float, float] = (10_000.0, 5_000.0, 2_000.0)
    layer_span: int = 1
    interference_channels: int = 0
    power_mw: float = 0.0


# ---------------------------------------------------------------------------
# Physical sizing constants
# ---------------------------------------------------------------------------

# A single polariton logic gate footprint [nm]
_GATE_XY_NM = 10_000.0  # 10 µm
_GATE_Z_NM = 2_000.0    # 2 µm layer height

# Number of elementary gates per multiply-accumulate operation
# (one Y-splitter + one interference zone + one detector ~ 3 gates)
_GATES_PER_MAC = 3

# Power per gate [mW] — derived from energy_per_gate (0.5 aJ) at 1 GHz ref clock:
#   P = 0.5e-18 J × 1e9 Hz = 5e-10 W = 5e-7 mW
_POWER_PER_GATE_MW = 5e-7


@dataclass
class PolaritonicMapping:
    """Complete mapping of a :class:`ComputeGraph` to polariton gate clusters.

    Attributes:
        model_name: Name of the source model.
        clusters: Gate clusters, one per compute node.
        total_gate_count: Sum of gates across all clusters.
        total_power_mw: Total estimated pump power.
    """

    model_name: str = "unnamed"
    clusters: list[GateCluster] = field(default_factory=list)

    @property
    def total_gate_count(self) -> int:
        return sum(c.gate_count for c in self.clusters)

    @property
    def total_power_mw(self) -> float:
        return sum(c.power_mw for c in self.clusters)

    @property
    def total_layer_span(self) -> int:
        return max((c.layer_span for c in self.clusters), default=1)


class PolaritonicMapper:
    """Translate a :class:`ComputeGraph` into :class:`PolaritonicMapping`.

    Strategy per operation type:

    * **MatMul** — mapped to an *Interference Grid*: each column of the
      weight matrix corresponds to a parallel waveguide channel.  Input
      amplitudes are encoded as laser pulse intensities; the interference
      pattern at the output plane yields the dot-product result.  This is
      analogous to an optical matrix-vector multiplier.

    * **ReLU / GELU** — mapped to a *Threshold Nonlinear* cluster:
      a bistable polariton condensate that transmits only above a pump
      threshold, naturally implementing a rectifying nonlinearity.

    * **Softmax / LayerNorm** — mapped to a *Normalisation Cascade*:
      a chain of coupled micro-ring resonators whose transmission
      collectively implements the normalisation denominator via
      evanescent feedback.

    * **Add / Mul** — mapped to an *Elementwise Coupler*: directional
      couplers that combine two waveguide fields additively (Add) or
      through intensity modulation (Mul).

    * **Embedding** — mapped to an *Embedding Hologram*: the weight
      table is stored in the holographic spin-crossover memory and
      recalled with a laser address pulse.

    * **Reshape / Transpose** — *Passthrough*: only routing changes,
      no physical gates required.
    """

    _CLUSTER_MAP: dict[OpType, ClusterType] = {
        OpType.MATMUL: ClusterType.INTERFERENCE_GRID,
        OpType.CONV2D: ClusterType.INTERFERENCE_GRID,
        OpType.RELU: ClusterType.THRESHOLD_NONLINEAR,
        OpType.GELU: ClusterType.THRESHOLD_NONLINEAR,
        OpType.SOFTMAX: ClusterType.NORMALISATION_CASCADE,
        OpType.LAYERNORM: ClusterType.NORMALISATION_CASCADE,
        OpType.ADD: ClusterType.ELEMENTWISE_COUPLER,
        OpType.MUL: ClusterType.ELEMENTWISE_COUPLER,
        OpType.EMBEDDING: ClusterType.EMBEDDING_HOLOGRAM,
        OpType.TRANSPOSE: ClusterType.PASSTHROUGH,
        OpType.RESHAPE: ClusterType.PASSTHROUGH,
        OpType.UNKNOWN: ClusterType.PASSTHROUGH,
    }

    def map(self, graph: ComputeGraph) -> PolaritonicMapping:
        """Map every node in *graph* to a polariton gate cluster."""
        mapping = PolaritonicMapping(model_name=graph.name)

        for node in graph.nodes:
            cluster = self._map_node(node)
            mapping.clusters.append(cluster)

        return mapping

    def _map_node(self, node: ComputeNode) -> GateCluster:
        ctype = self._CLUSTER_MAP.get(node.op_type, ClusterType.PASSTHROUGH)
        gc = GateCluster(
            cluster_id=f"cl_{node.node_id}",
            cluster_type=ctype,
            source_node_id=node.node_id,
        )

        if ctype == ClusterType.INTERFERENCE_GRID:
            self._size_interference_grid(node, gc)
        elif ctype == ClusterType.THRESHOLD_NONLINEAR:
            self._size_threshold(node, gc)
        elif ctype == ClusterType.NORMALISATION_CASCADE:
            self._size_normalisation(node, gc)
        elif ctype == ClusterType.ELEMENTWISE_COUPLER:
            self._size_elementwise(node, gc)
        elif ctype == ClusterType.EMBEDDING_HOLOGRAM:
            self._size_embedding(node, gc)
        # PASSTHROUGH → 0 gates

        gc.power_mw = gc.gate_count * _POWER_PER_GATE_MW
        return gc

    # ---- sizing helpers ----

    def _size_interference_grid(self, node: ComputeNode, gc: GateCluster) -> None:
        """MatMul / Conv2D → parallel interference channels.

        For (M, K) × (K, N): we need N parallel interference channels,
        each with K beam-splitters, so gate_count ≈ K*N * _GATES_PER_MAC.
        The grid spans ceil(sqrt(N)) columns in x and K rows in z.
        """
        if len(node.input_shapes) < 2:
            gc.gate_count = _GATES_PER_MAC
            gc.interference_channels = 1
            gc.layer_span = 1
            return

        a, b = node.input_shapes[0], node.input_shapes[1]
        k = a.dims[-1] if a.ndim >= 1 else 1
        n = b.dims[-1] if b.ndim >= 2 else 1
        m = a.dims[-2] if a.ndim >= 2 else 1

        # Total MAC operations
        macs = m * k * n
        gc.gate_count = macs * _GATES_PER_MAC
        gc.interference_channels = n

        # Layers: each row of K multipliers stacks vertically
        gc.layer_span = min(k, 64)  # cap at 64 layers per cluster

        # Footprint: channels laid out in x, accumulation depth in y
        import math
        cols = int(math.ceil(math.sqrt(n)))
        gc.footprint_nm = (
            cols * _GATE_XY_NM,
            int(math.ceil(n / max(cols, 1))) * _GATE_XY_NM,
            gc.layer_span * _GATE_Z_NM,
        )

    def _size_threshold(self, node: ComputeNode, gc: GateCluster) -> None:
        """ReLU / GELU → one bistable switch per element."""
        numel = node.output_shape.numel
        gc.gate_count = numel  # one gate per element
        gc.layer_span = 1
        import math
        side = int(math.ceil(math.sqrt(numel)))
        gc.footprint_nm = (side * _GATE_XY_NM, side * _GATE_XY_NM, _GATE_Z_NM)

    def _size_normalisation(self, node: ComputeNode, gc: GateCluster) -> None:
        """Softmax / LayerNorm → coupled resonator chain."""
        numel = node.output_shape.numel
        # Each element needs ~3 gates (ring resonator + feedback + readout)
        gc.gate_count = numel * 3
        gc.layer_span = 2  # feedback loop spans two layers
        import math
        side = int(math.ceil(math.sqrt(numel)))
        gc.footprint_nm = (side * _GATE_XY_NM, side * _GATE_XY_NM, 2 * _GATE_Z_NM)

    def _size_elementwise(self, node: ComputeNode, gc: GateCluster) -> None:
        """Add / Mul → one coupler per element."""
        numel = node.output_shape.numel
        gc.gate_count = numel
        gc.layer_span = 1
        import math
        side = int(math.ceil(math.sqrt(numel)))
        gc.footprint_nm = (side * _GATE_XY_NM, side * _GATE_XY_NM, _GATE_Z_NM)

    def _size_embedding(self, node: ComputeNode, gc: GateCluster) -> None:
        """Embedding → holographic recall gates."""
        numel = node.output_shape.numel
        # One holographic read gate per output element
        gc.gate_count = numel
        gc.layer_span = 1
        import math
        side = int(math.ceil(math.sqrt(numel)))
        gc.footprint_nm = (side * _GATE_XY_NM, side * _GATE_XY_NM, _GATE_Z_NM)
