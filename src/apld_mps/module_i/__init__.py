"""Module I — AI Tensor Compiler.

Bridges high-level neural-network models (ONNX, Transformer specs) to the
Aethel polariton logic architecture by:

1. **Model Parser** — extracts computation graphs from ONNX or Transformer
   specifications, identifying MatMul, ReLU, Softmax and other operations.

2. **Polaritonic Mapping** — translates tensor operations into specialised
   gate clusters (interference grids, bistable switches, resonator cascades)
   that exploit wave-interference in the sapphire lattice.

3. **Memory Interface** — manages holographic spin-crossover (SCO) memory
   for weight storage, enabling laser-read pulses to flow directly into
   compute layers without a classical bus.

4. **Resource Estimator** — computes the required number of PlacedGate3D,
   z-layers, monolith dimensions, and power budget for a given model size.

5. **Latency Analyser** — simulates end-to-end light-pulse propagation
   through all Transformer layers and reports theoretical tokens/s.
"""

from .model_parser import (
    OpType,
    TensorShape,
    ComputeNode,
    ComputeEdge,
    ComputeGraph,
    ModelParser,
)
from .polaritonic_mapping import (
    ClusterType,
    GateCluster,
    PolaritonicMapping,
    PolaritonicMapper,
)
from .memory_interface import (
    SCOCellSpec,
    WeightBlock,
    MemoryRegion,
    ReadPulseResult,
    HolographicMemoryInterface,
)
from .resource_estimator import (
    ResourceEstimate,
    ResourceEstimator,
)
from .latency_analyser import (
    LayerLatency,
    LatencyReport,
    LatencyAnalyser,
)

__all__ = [
    "OpType",
    "TensorShape",
    "ComputeNode",
    "ComputeEdge",
    "ComputeGraph",
    "ModelParser",
    "ClusterType",
    "GateCluster",
    "PolaritonicMapping",
    "PolaritonicMapper",
    "SCOCellSpec",
    "WeightBlock",
    "MemoryRegion",
    "ReadPulseResult",
    "HolographicMemoryInterface",
    "ResourceEstimate",
    "ResourceEstimator",
    "LayerLatency",
    "LatencyReport",
    "LatencyAnalyser",
]
