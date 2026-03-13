"""Resource Estimator — compute physical requirements for AI model deployment.

Given a model size and its polaritonic mapping, estimates the required
number of :class:`PlacedGate3D` instances, sapphire monolith dimensions,
layer count, memory regions, and power budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .model_parser import ComputeGraph
from .polaritonic_mapping import (
    PolaritonicMapper,
    PolaritonicMapping,
    _GATE_XY_NM,
    _GATE_Z_NM,
)
from .memory_interface import HolographicMemoryInterface, SCOCellSpec


@dataclass
class ResourceEstimate:
    """Complete resource estimate for deploying an AI model on the Aethel chip.

    Attributes:
        model_name: Source model name.
        total_parameters: Number of trainable parameters.
        total_flops_per_token: FLOPs per forward pass (one token).
        total_gate_count: Number of polariton gates required.
        layer_count: Number of z-layers in the sapphire monolith.
        monolith_volume_nm: Required (x, y, z) sapphire volume [nm].
        monolith_volume_mm: Same in millimetres for readability.
        memory_regions: Number of holographic memory regions.
        memory_capacity_bytes: Total weight storage capacity [bytes].
        total_power_mw: Estimated total optical pump power [mW].
        total_power_w: Same in watts.
        weight_bits: Quantisation precision.
        weight_storage_bytes: Actual bytes needed for weights.
    """

    model_name: str = "unnamed"
    total_parameters: int = 0
    total_flops_per_token: int = 0
    total_gate_count: int = 0
    layer_count: int = 0
    monolith_volume_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    monolith_volume_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    memory_regions: int = 0
    memory_capacity_bytes: int = 0
    total_power_mw: float = 0.0
    total_power_w: float = 0.0
    weight_bits: int = 8
    weight_storage_bytes: int = 0

    def summary(self) -> str:
        """Human-readable summary of the resource estimate."""
        lines = [
            f"=== Resource Estimate: {self.model_name} ===",
            f"Parameters:        {self.total_parameters:>15,}",
            f"FLOPs/token:       {self.total_flops_per_token:>15,}",
            f"Polariton gates:   {self.total_gate_count:>15,}",
            f"Z-layers:          {self.layer_count:>15,}",
            f"Monolith (mm):     {self.monolith_volume_mm[0]:.1f} x "
            f"{self.monolith_volume_mm[1]:.1f} x "
            f"{self.monolith_volume_mm[2]:.1f}",
            f"Memory regions:    {self.memory_regions:>15,}",
            f"Memory capacity:   {self.memory_capacity_bytes:>15,} bytes",
            f"Weight storage:    {self.weight_storage_bytes:>15,} bytes",
            f"Weight precision:  {self.weight_bits:>15} bit",
            f"Optical power:     {self.total_power_w:>15.2f} W",
        ]
        return "\n".join(lines)


class ResourceEstimator:
    """Estimate physical resources for deploying a neural network.

    Combines the polaritonic mapping (gate count, footprints) with the
    holographic memory interface (weight storage) to produce a complete
    :class:`ResourceEstimate`.

    Parameters:
        cell_spec: SCO memory cell specification.
        weight_bits: Default weight quantisation precision.
        memory_region_volume_nm: Default memory region volume.
        packing_efficiency: How tightly clusters pack (0–1). Lower
            values leave more room for routing waveguides.
    """

    def __init__(
        self,
        cell_spec: SCOCellSpec | None = None,
        weight_bits: int = 8,
        memory_region_volume_nm: tuple[float, float, float] = (
            1_000_000.0, 1_000_000.0, 100_000.0,
        ),
        packing_efficiency: float = 0.5,
    ) -> None:
        self.cell_spec = cell_spec or SCOCellSpec()
        self.weight_bits = weight_bits
        self.memory_region_volume_nm = memory_region_volume_nm
        self.packing_efficiency = packing_efficiency

    def estimate(
        self,
        graph: ComputeGraph,
        mapping: PolaritonicMapping | None = None,
    ) -> ResourceEstimate:
        """Produce a full resource estimate.

        Parameters:
            graph: The parsed computation graph.
            mapping: Pre-computed polaritonic mapping (computed if None).
        """
        if mapping is None:
            mapping = PolaritonicMapper().map(graph)

        total_gates = mapping.total_gate_count
        total_power = mapping.total_power_mw

        # --- Monolith volume ---
        # XY: pack gates in a square grid
        gate_area = _GATE_XY_NM * _GATE_XY_NM
        total_area = total_gates * gate_area / self.packing_efficiency
        side_nm = math.sqrt(total_area)

        # Z: number of layers from the mapping
        n_layers = max(mapping.total_layer_span, 1)
        z_nm = n_layers * _GATE_Z_NM

        volume_nm = (side_nm, side_nm, z_nm)
        nm_to_mm = 1e-6
        volume_mm = tuple(v * nm_to_mm for v in volume_nm)

        # --- Memory ---
        mem = HolographicMemoryInterface(cell_spec=self.cell_spec)
        weight_bytes = math.ceil(graph.total_parameters * self.weight_bits / 8)
        regions = mem.allocate_model_weights(
            graph.total_parameters,
            bits_per_weight=self.weight_bits,
            region_volume_nm=self.memory_region_volume_nm,
        )
        mem_capacity = mem.total_capacity_bytes()

        return ResourceEstimate(
            model_name=graph.name,
            total_parameters=graph.total_parameters,
            total_flops_per_token=graph.total_flops,
            total_gate_count=total_gates,
            layer_count=n_layers,
            monolith_volume_nm=volume_nm,
            monolith_volume_mm=volume_mm,  # type: ignore[arg-type]
            memory_regions=len(regions),
            memory_capacity_bytes=mem_capacity,
            total_power_mw=total_power,
            total_power_w=total_power / 1000.0,
            weight_bits=self.weight_bits,
            weight_storage_bytes=weight_bytes,
        )

    def estimate_from_spec(
        self,
        *,
        name: str = "model",
        n_layers: int = 32,
        d_model: int = 4096,
        n_heads: int = 32,
        d_ff: int | None = None,
        vocab_size: int = 32000,
        seq_len: int = 2048,
    ) -> ResourceEstimate:
        """Convenience: build graph from Transformer spec and estimate."""
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
        return self.estimate(graph)
