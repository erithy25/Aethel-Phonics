"""Detail 2.3 — Weight & Holographic Memory Loader.

Loads neural-network weights from standard formats (``.safetensors``,
``.gguf``, ``.npz``, or raw NumPy arrays) and *simulates* storage and
retrieval through Module I's ``HolographicMemoryInterface``.

Each weight read consumes simulated time:
    t_read = read_time_ps + propagation_distance / (c / n_sapphire)

The loader tracks cumulative read latency and energy so that the AFEE
engine can report realistic I/O costs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np

from ..constants import C_LIGHT
from ..module_i.memory_interface import (
    HolographicMemoryInterface,
    SCOCellSpec,
)


@dataclass
class WeightTensor:
    """A named weight tensor loaded into holographic memory.

    Attributes:
        name: Layer/parameter name (e.g. ``"layers.0.attn.wq"``).
        shape: Tensor shape.
        dtype: NumPy dtype.
        data: The actual weight data as a NumPy array.
        read_latency_ps: Simulated holographic read time [ps].
        read_energy_pj: Simulated read energy [pJ].
        region_id: SCO memory region where the tensor is stored.
    """

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    data: np.ndarray
    read_latency_ps: float = 0.0
    read_energy_pj: float = 0.0
    region_id: str = ""


@dataclass
class LoadReport:
    """Summary of a weight-loading session.

    Attributes:
        n_tensors: Number of weight tensors loaded.
        total_parameters: Total scalar parameter count.
        total_bytes: Total storage consumed [bytes].
        total_read_latency_ps: Cumulative holographic read time [ps].
        total_read_energy_pj: Cumulative read energy [pJ].
        memory_utilisation: Fraction of allocated SCO capacity used.
        source_format: File format or 'numpy' for in-memory weights.
    """

    n_tensors: int = 0
    total_parameters: int = 0
    total_bytes: int = 0
    total_read_latency_ps: float = 0.0
    total_read_energy_pj: float = 0.0
    memory_utilisation: float = 0.0
    source_format: str = "numpy"


class WeightLoader:
    """Loads and serves model weights through simulated holographic memory.

    Parameters:
        cell_spec: SCO cell specification (pitch, energies, timings).
        propagation_distance_nm: Mean distance from memory region to
            compute layer [nm].
        n_sapphire: Refractive index of sapphire at operating wavelength.
    """

    def __init__(
        self,
        cell_spec: SCOCellSpec | None = None,
        propagation_distance_nm: float = 50_000.0,
        n_sapphire: float = 1.77,
    ) -> None:
        self._mem = HolographicMemoryInterface(
            n_sapphire=n_sapphire,
            cell_spec=cell_spec,
        )
        self._propagation_nm = propagation_distance_nm
        self._weights: dict[str, WeightTensor] = {}
        self._report = LoadReport()

    @property
    def report(self) -> LoadReport:
        return self._report

    @property
    def weight_names(self) -> list[str]:
        return list(self._weights.keys())

    def get(self, name: str) -> WeightTensor:
        """Retrieve a loaded weight tensor by name."""
        return self._weights[name]

    # ------------------------------------------------------------------
    # Loading from various sources
    # ------------------------------------------------------------------

    def load_numpy(self, weights: dict[str, np.ndarray]) -> LoadReport:
        """Load weights from a dict of name → ndarray.

        Each tensor is allocated in a fresh SCO memory region and a
        holographic read is simulated.
        """
        self._weights.clear()
        self._report = LoadReport(source_format="numpy")

        for name, arr in weights.items():
            wt = self._store_and_read(name, arr)
            self._weights[name] = wt

        self._finalise_report()
        return self._report

    def load_npz(self, path: str | Path) -> LoadReport:
        """Load weights from a ``.npz`` file."""
        data = np.load(str(path))
        return self.load_numpy(dict(data))

    def load_safetensors(self, path: str | Path) -> LoadReport:
        """Load weights from a ``.safetensors`` file.

        Requires the ``safetensors`` package.  Falls back to a stub if
        not installed.
        """
        try:
            from safetensors.numpy import load_file
            tensors = load_file(str(path))
        except ImportError:
            tensors = self._stub_load(path, "safetensors")
        return self.load_numpy(tensors)

    def load_gguf(self, path: str | Path) -> LoadReport:
        """Load weights from a ``.gguf`` file.

        Requires the ``gguf`` package.  Falls back to a stub if not installed.
        """
        try:
            from gguf import GGUFReader
            reader = GGUFReader(str(path))
            tensors = {}
            for tensor in reader.tensors:
                tensors[tensor.name] = np.array(tensor.data)
        except ImportError:
            tensors = self._stub_load(path, "gguf")
        return self.load_numpy(tensors)

    def load_auto(self, path: str | Path) -> LoadReport:
        """Auto-detect format from file extension and load."""
        p = Path(path)
        ext = p.suffix.lower()
        if ext == ".npz":
            return self.load_npz(p)
        if ext == ".safetensors":
            return self.load_safetensors(p)
        if ext in (".gguf",):
            return self.load_gguf(p)
        raise ValueError(f"Unsupported weight format: {ext}")

    # ------------------------------------------------------------------
    # Synthetic weight generation (for testing / demo)
    # ------------------------------------------------------------------

    def generate_random_transformer(
        self,
        *,
        n_layers: int = 4,
        d_model: int = 64,
        d_ff: int = 256,
        n_heads: int = 4,
        vocab_size: int = 1000,
        seed: int = 42,
    ) -> LoadReport:
        """Generate random transformer weights and load them.

        Useful for testing the full pipeline without a real model file.
        """
        rng = np.random.default_rng(seed)
        scale = 0.02
        weights: dict[str, np.ndarray] = {}

        # Embedding
        weights["embed.weight"] = rng.normal(0, scale, (vocab_size, d_model)).astype(np.float32)

        d_head = d_model // n_heads
        for i in range(n_layers):
            pre = f"layers.{i}"
            # Attention
            weights[f"{pre}.attn.wq"] = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
            weights[f"{pre}.attn.wk"] = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
            weights[f"{pre}.attn.wv"] = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
            weights[f"{pre}.attn.wo"] = rng.normal(0, scale, (d_model, d_model)).astype(np.float32)
            # FFN (SwiGLU)
            weights[f"{pre}.ffn.w1"] = rng.normal(0, scale, (d_model, d_ff)).astype(np.float32)
            weights[f"{pre}.ffn.w2"] = rng.normal(0, scale, (d_ff, d_model)).astype(np.float32)
            weights[f"{pre}.ffn.w3"] = rng.normal(0, scale, (d_model, d_ff)).astype(np.float32)
            # Norms
            weights[f"{pre}.attn_norm"] = np.ones(d_model, dtype=np.float32)
            weights[f"{pre}.ffn_norm"] = np.ones(d_model, dtype=np.float32)

        # Output
        weights["output_norm"] = np.ones(d_model, dtype=np.float32)
        weights["output.weight"] = rng.normal(0, scale, (d_model, vocab_size)).astype(np.float32)

        return self.load_numpy(weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_and_read(self, name: str, arr: np.ndarray) -> WeightTensor:
        """Allocate an SCO region, store the tensor, simulate a read."""
        rows = arr.shape[0] if arr.ndim >= 2 else 1
        cols = arr.shape[1] if arr.ndim >= 2 else arr.size
        bits = 8  # INT8 quantised

        # Allocate a region sized to this tensor
        n_bytes = arr.size * (bits // 8)
        # Region volume: pack cells at 50 nm pitch
        cell_pitch = self._mem.cell_spec.cell_pitch_nm
        cells_needed = n_bytes * 8  # bits
        side_cells = max(1, int(np.cbrt(cells_needed)) + 1)
        side_nm = side_cells * cell_pitch
        vol = (side_nm, side_nm, side_nm)

        region = self._mem.allocate_region(f"region-{name}", volume_nm=vol)
        block = self._mem.store_weights(region, name, rows, cols, bits_per_weight=bits)
        read_result = self._mem.read_weights(
            region, name, propagation_distance_nm=self._propagation_nm,
        )

        self._report.n_tensors += 1
        self._report.total_parameters += arr.size
        self._report.total_bytes += n_bytes
        self._report.total_read_latency_ps += read_result.total_latency_ps
        self._report.total_read_energy_pj += read_result.read_energy_pj

        return WeightTensor(
            name=name,
            shape=arr.shape,
            dtype=arr.dtype,
            data=arr,
            read_latency_ps=read_result.total_latency_ps,
            read_energy_pj=read_result.read_energy_pj,
            region_id=region.region_id,
        )

    def _finalise_report(self) -> None:
        """Compute aggregate metrics after all tensors are loaded."""
        total_cap = self._mem.total_capacity_bytes()
        if total_cap > 0:
            self._report.memory_utilisation = self._report.total_bytes / total_cap

    @staticmethod
    def _stub_load(path: str | Path, fmt: str) -> dict[str, np.ndarray]:
        """Stub: generate a single small tensor when the real loader is unavailable."""
        return {f"stub_{fmt}": np.zeros((1,), dtype=np.float32)}
