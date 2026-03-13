"""Holographic Spin-Crossover Memory Interface.

Models the storage and retrieval of neural-network weight matrices using
a holographic memory encoded in spin-crossover (SCO) molecular arrays
embedded within the sapphire lattice.

Physical principle
------------------
Spin-crossover molecules (e.g. Fe(II) complexes) exhibit bistable
high-spin / low-spin states that can be switched by femtosecond laser
pulses.  A 3D holographic pattern of HS/LS states encodes a weight
matrix; reading is performed by a low-intensity probe pulse whose
diffraction pattern reconstructs the stored matrix row.

The key advantage over a classical memory bus is that the read-out
pulse propagates directly through the computation layers, feeding
weights into the interference grid without electronic intermediaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..constants import C_LIGHT, NM_TO_M, PS_TO_S


@dataclass
class SCOCellSpec:
    """Specification of a single spin-crossover memory cell.

    Attributes:
        cell_pitch_nm: Centre-to-centre spacing of SCO molecules [nm].
        switch_energy_fj: Energy to flip one molecule [fJ].
        read_energy_fj: Energy of a probe pulse to read one cell [fJ].
        retention_time_us: How long the spin state is stable [µs].
        switch_time_ps: Time to flip one cell [ps].
        read_time_ps: Time to read one cell [ps].
    """

    cell_pitch_nm: float = 50.0       # ~50 nm pitch (sub-wavelength)
    switch_energy_fj: float = 10.0    # 10 fJ per write
    read_energy_fj: float = 1.0       # 1 fJ per read
    retention_time_us: float = 1e6    # effectively permanent at room temp
    switch_time_ps: float = 0.5       # 500 fs switching
    read_time_ps: float = 0.1        # 100 fs read


@dataclass
class WeightBlock:
    """A contiguous block of model weights stored holographically.

    Attributes:
        block_id: Unique identifier.
        rows: Number of rows in the weight matrix.
        cols: Number of columns.
        bits_per_weight: Precision of each weight (default 8-bit quantised).
        layer_index: Which model layer this block belongs to.
    """

    block_id: str
    rows: int
    cols: int
    bits_per_weight: int = 8
    layer_index: int = 0

    @property
    def total_weights(self) -> int:
        return self.rows * self.cols

    @property
    def total_bits(self) -> int:
        return self.total_weights * self.bits_per_weight

    @property
    def size_bytes(self) -> int:
        return math.ceil(self.total_bits / 8)


@dataclass
class MemoryRegion:
    """A 3D region of the sapphire monolith dedicated to weight storage.

    Attributes:
        region_id: Unique identifier.
        volume_nm: (x, y, z) dimensions [nm].
        cell_spec: SCO cell specification.
        weight_blocks: Stored weight blocks.
    """

    region_id: str
    volume_nm: tuple[float, float, float] = (100_000.0, 100_000.0, 10_000.0)
    cell_spec: SCOCellSpec = field(default_factory=SCOCellSpec)
    weight_blocks: list[WeightBlock] = field(default_factory=list)

    @property
    def capacity_cells(self) -> int:
        """Maximum number of SCO cells that fit in this region."""
        pitch = self.cell_spec.cell_pitch_nm
        nx = int(self.volume_nm[0] / pitch)
        ny = int(self.volume_nm[1] / pitch)
        nz = int(self.volume_nm[2] / pitch)
        return nx * ny * nz

    @property
    def capacity_bits(self) -> int:
        """Total bit capacity (1 bit per SCO cell)."""
        return self.capacity_cells

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_bits // 8

    @property
    def utilisation(self) -> float:
        """Fraction of capacity used by stored weight blocks."""
        used = sum(wb.total_bits for wb in self.weight_blocks)
        cap = self.capacity_bits
        return used / cap if cap > 0 else 0.0


@dataclass
class ReadPulseResult:
    """Result of a holographic weight read operation.

    Attributes:
        block_id: Which weight block was read.
        read_time_ps: Total time to read all weights [ps].
        read_energy_pj: Total energy consumed [pJ].
        propagation_distance_nm: Distance the read pulse travels to
            reach the first computation layer [nm].
        propagation_time_ps: Light travel time for that distance [ps].
    """

    block_id: str
    read_time_ps: float = 0.0
    read_energy_pj: float = 0.0
    propagation_distance_nm: float = 0.0
    propagation_time_ps: float = 0.0

    @property
    def total_latency_ps(self) -> float:
        """Read time plus propagation to compute layer."""
        return self.read_time_ps + self.propagation_time_ps


class HolographicMemoryInterface:
    """Interface between the holographic SCO memory and the compute fabric.

    This class manages weight storage, read-pulse scheduling, and the
    direct optical path from memory regions to interference-grid layers
    (bypassing any electronic bus).

    Parameters:
        n_sapphire: Refractive index of the sapphire substrate at the
            operating wavelength (default 1.77 at ~720 nm).
        cell_spec: Default SCO cell specification.
    """

    def __init__(
        self,
        n_sapphire: float = 1.77,
        cell_spec: SCOCellSpec | None = None,
    ) -> None:
        self.n_sapphire = n_sapphire
        self.cell_spec = cell_spec or SCOCellSpec()
        self.regions: list[MemoryRegion] = []

    def allocate_region(
        self,
        region_id: str,
        volume_nm: tuple[float, float, float] = (100_000.0, 100_000.0, 10_000.0),
    ) -> MemoryRegion:
        """Allocate a new memory region in the sapphire volume."""
        region = MemoryRegion(
            region_id=region_id,
            volume_nm=volume_nm,
            cell_spec=self.cell_spec,
        )
        self.regions.append(region)
        return region

    def store_weights(
        self,
        region: MemoryRegion,
        block_id: str,
        rows: int,
        cols: int,
        bits_per_weight: int = 8,
        layer_index: int = 0,
    ) -> WeightBlock:
        """Store a weight matrix block in a memory region.

        Raises:
            ValueError: If the region has insufficient capacity.
        """
        wb = WeightBlock(
            block_id=block_id,
            rows=rows,
            cols=cols,
            bits_per_weight=bits_per_weight,
            layer_index=layer_index,
        )
        used = sum(b.total_bits for b in region.weight_blocks)
        if used + wb.total_bits > region.capacity_bits:
            raise ValueError(
                f"Region '{region.region_id}' capacity exceeded: "
                f"need {wb.total_bits} bits, only {region.capacity_bits - used} free"
            )
        region.weight_blocks.append(wb)
        return wb

    def read_weights(
        self,
        region: MemoryRegion,
        block_id: str,
        propagation_distance_nm: float = 0.0,
    ) -> ReadPulseResult:
        """Simulate a holographic read pulse for a weight block.

        The read time is the single-cell read time (all cells are read
        in parallel by the holographic diffraction pattern).  The
        propagation time models the optical path from the memory region
        to the nearest computation layer.

        Parameters:
            region: The memory region containing the block.
            block_id: ID of the weight block to read.
            propagation_distance_nm: Distance to the compute layer [nm].
        """
        wb = None
        for b in region.weight_blocks:
            if b.block_id == block_id:
                wb = b
                break
        if wb is None:
            raise KeyError(f"Weight block '{block_id}' not found in region '{region.region_id}'")

        # Holographic read: all weights in parallel, time = single cell read
        read_time = self.cell_spec.read_time_ps

        # Energy: one probe pulse covers the entire hologram
        read_energy_pj = self.cell_spec.read_energy_fj * 1e-3  # fJ → pJ (per cell is 1fJ, but holographic = 1 pulse)

        # Propagation time: distance / (c / n)
        v_medium = C_LIGHT / self.n_sapphire  # m/s
        dist_m = propagation_distance_nm * NM_TO_M
        prop_time_s = dist_m / v_medium
        prop_time_ps = prop_time_s / PS_TO_S

        return ReadPulseResult(
            block_id=block_id,
            read_time_ps=read_time,
            read_energy_pj=read_energy_pj,
            propagation_distance_nm=propagation_distance_nm,
            propagation_time_ps=prop_time_ps,
        )

    def total_capacity_bytes(self) -> int:
        """Total byte capacity across all allocated regions."""
        return sum(r.capacity_bytes for r in self.regions)

    def allocate_model_weights(
        self,
        total_parameters: int,
        bits_per_weight: int = 8,
        region_volume_nm: tuple[float, float, float] = (1_000_000.0, 1_000_000.0, 100_000.0),
    ) -> list[MemoryRegion]:
        """Auto-allocate enough memory regions to hold all model weights.

        Parameters:
            total_parameters: Number of weights to store.
            bits_per_weight: Quantisation precision.
            region_volume_nm: Size of each memory region.

        Returns:
            List of allocated memory regions.
        """
        total_bits = total_parameters * bits_per_weight
        sample_region = MemoryRegion(
            region_id="_sample", volume_nm=region_volume_nm, cell_spec=self.cell_spec
        )
        bits_per_region = sample_region.capacity_bits
        if bits_per_region <= 0:
            raise ValueError("Region volume too small for any cells")

        n_regions = math.ceil(total_bits / bits_per_region)
        regions = []
        remaining = total_parameters
        for i in range(n_regions):
            region = self.allocate_region(f"weight_region_{i}", region_volume_nm)
            # Fill this region
            chunk = min(remaining, bits_per_region // bits_per_weight)
            if chunk > 0:
                self.store_weights(
                    region,
                    block_id=f"weights_{i}",
                    rows=1,
                    cols=chunk,
                    bits_per_weight=bits_per_weight,
                    layer_index=i,
                )
            remaining -= chunk
            regions.append(region)

        return regions
