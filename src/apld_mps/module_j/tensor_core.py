"""Detail 2.1 — Polariton Tensor Core Emulation.

Standard tensor operations (MatMul, VectorAdd, ReLU / SiLU) backed by NumPy,
with a physics-coupled noise-injection layer.

Physics coupling:
    The core queries a *coherence map* derived from Module G's
    ``SelfHealingRouter.availability`` grid.  When the mean coherence of the
    sector responsible for a given tensor tile drops below a configurable
    threshold (default 0.8), Gaussian noise scaled by the coherence deficit
    is added to the result.  This faithfully simulates how cosmic-ray–induced
    decoherence propagates into AI inference outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoherenceSector:
    """A single sector of the monolith with a coherence level.

    Attributes:
        sector_id: Unique identifier (grid index tuple or string).
        coherence: Mean coherence in [0, 1].  1.0 = perfect, 0.0 = fully
                   disrupted.
    """

    sector_id: str
    coherence: float = 1.0


class PolaritonicTensorCore:
    """Tensor compute unit that emulates polariton-gate arithmetic.

    All operations accept and return real-valued ``np.ndarray`` tensors.
    After each operation the result is optionally corrupted by Gaussian noise
    whose variance is proportional to the local coherence deficit.

    Parameters:
        coherence_threshold: Coherence level below which noise is injected
            (default 0.8).
        noise_scale: Base standard-deviation of injected noise at zero
            coherence (default 0.01).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        coherence_threshold: float = 0.8,
        noise_scale: float = 0.01,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= coherence_threshold <= 1.0:
            raise ValueError("coherence_threshold must be in [0, 1]")
        self.coherence_threshold = coherence_threshold
        self.noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)

        # The sector coherence map is set externally (e.g. from Module G).
        self._sector_coherence: float = 1.0

        # Counters
        self._ops_total: int = 0
        self._ops_noisy: int = 0

    # ------------------------------------------------------------------
    # Coherence interface
    # ------------------------------------------------------------------

    def set_sector_coherence(self, coherence: float) -> None:
        """Update the coherence level for the current compute sector."""
        self._sector_coherence = max(0.0, min(1.0, coherence))

    def set_availability_map(self, availability: np.ndarray) -> None:
        """Set sector coherence from a Module-G availability grid.

        Takes the *mean* availability as the sector coherence.
        """
        self._sector_coherence = float(np.mean(availability))

    @property
    def sector_coherence(self) -> float:
        return self._sector_coherence

    @property
    def ops_total(self) -> int:
        return self._ops_total

    @property
    def ops_noisy(self) -> int:
        return self._ops_noisy

    # ------------------------------------------------------------------
    # Noise injection
    # ------------------------------------------------------------------

    def _maybe_inject_noise(self, result: np.ndarray) -> np.ndarray:
        """Add coherence-dependent Gaussian noise if below threshold."""
        self._ops_total += 1
        if self._sector_coherence < self.coherence_threshold:
            deficit = 1.0 - self._sector_coherence
            sigma = self.noise_scale * deficit
            noise = self._rng.normal(0.0, sigma, size=result.shape)
            self._ops_noisy += 1
            return result + noise
        return result

    # ------------------------------------------------------------------
    # Core tensor operations
    # ------------------------------------------------------------------

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication with coherence-based noise injection."""
        result = a @ b
        return self._maybe_inject_noise(result)

    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise addition with coherence-based noise injection."""
        result = a + b
        return self._maybe_inject_noise(result)

    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x)."""
        result = np.maximum(0.0, x)
        return self._maybe_inject_noise(result)

    def silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU (Swish) activation: x * sigmoid(x)."""
        result = x * (1.0 / (1.0 + np.exp(-x)))
        return self._maybe_inject_noise(result)

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax along *axis*."""
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(shifted)
        result = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        return self._maybe_inject_noise(result)

    def layer_norm(
        self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5,
    ) -> np.ndarray:
        """Layer normalisation over the last axis."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(var + eps)
        result = gamma * normed + beta
        return self._maybe_inject_noise(result)

    def rms_norm(
        self, x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5,
    ) -> np.ndarray:
        """RMSNorm (used by Llama): gamma * x / rms(x)."""
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        result = gamma * x / rms
        return self._maybe_inject_noise(result)
