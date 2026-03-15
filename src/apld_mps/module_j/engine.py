"""Aethel Functional Emulation Engine (AFEE).

The main inference engine that runs a small transformer model through the
full Aethel physics stack.  Every tensor operation is modulated by:

* **Cosmic-ray noise** — ``PolaritonicTensorCore`` (Module G coupling)
* **Thermal drift** — ``ThermalDriftLayer`` (Module F coupling)
* **Reversible compute** — ``ReversibleInferenceEngine`` (Module D gates)
* **Holographic memory** — ``WeightLoader`` (Module I coupling)

The engine exposes a simple ``forward(token_ids) → logits`` interface and a
``generate(prompt_ids, max_tokens) → token_ids`` auto-regressive loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from ..module_f.heat_optics import ThermoOpticConfig
from .tensor_core import PolaritonicTensorCore
from .reversible_inference import ReversibleInferenceEngine, ReversibilityReport
from .weight_loader import WeightLoader, LoadReport
from .thermal_drift import ThermalDriftLayer, DriftSnapshot


class InferenceMode(Enum):
    """Compute mode selector."""
    STANDARD = auto()       # Noise injection only
    REVERSIBLE = auto()     # Reversible gates (zero Landauer heat)
    PHYSICS_FULL = auto()   # Full physics coupling (noise + thermal drift)


@dataclass
class AFEEConfig:
    """Configuration for the AFEE engine.

    Attributes:
        n_layers: Number of transformer layers.
        d_model: Hidden dimension.
        d_ff: Feed-forward dimension.
        n_heads: Number of attention heads.
        vocab_size: Vocabulary size.
        mode: Inference mode.
        coherence_threshold: Noise injection threshold (Module G).
        noise_scale: Base noise standard deviation.
        breakdown_temperature_K: Thermal breakdown threshold.
        reversible_fraction: Fraction of reversible gates [0, 1].
        seed: Random seed.
    """

    n_layers: int = 4
    d_model: int = 64
    d_ff: int = 256
    n_heads: int = 4
    vocab_size: int = 1000
    mode: InferenceMode = InferenceMode.PHYSICS_FULL
    coherence_threshold: float = 0.8
    noise_scale: float = 0.01
    breakdown_temperature_K: float = 409.0
    reversible_fraction: float = 0.0
    seed: int = 42


@dataclass
class FidelityReport:
    """Live fidelity assessment of the inference output.

    Attributes:
        thermal_fidelity: MZI contrast from thermo-optic phase error [0, 1].
        coherence_fidelity: Fraction of ops without cosmic-ray noise [0, 1].
        reversibility_fidelity: Fraction of ops that were reversible [0, 1].
        overall_fidelity: Product of all fidelity scores [0, 1].
        is_breakdown: Whether thermal breakdown has occurred.
        peak_temperature_K: Current peak chip temperature.
        phase_error_rad: Current thermo-optic phase error.
    """

    thermal_fidelity: float = 1.0
    coherence_fidelity: float = 1.0
    reversibility_fidelity: float = 1.0
    overall_fidelity: float = 1.0
    is_breakdown: bool = False
    peak_temperature_K: float = 300.0
    phase_error_rad: float = 0.0


@dataclass
class InferenceResult:
    """Result of a single forward pass.

    Attributes:
        logits: Output logits of shape (seq_len, vocab_size).
        fidelity: Fidelity report.
        load_report: Weight-loading statistics.
        reversibility: Reversibility audit.
        thermal_history: Per-op thermal snapshots.
        latency_ps: Estimated total latency [ps].
    """

    logits: np.ndarray
    fidelity: FidelityReport
    load_report: LoadReport
    reversibility: ReversibilityReport
    thermal_history: list[DriftSnapshot] = field(default_factory=list)
    latency_ps: float = 0.0


class AFEE:
    """Aethel Functional Emulation Engine.

    Usage::

        cfg = AFEEConfig(n_layers=4, d_model=64, d_ff=256)
        engine = AFEE(cfg)
        engine.load_weights()          # generate random weights
        result = engine.forward([1, 42, 7])
        print(result.fidelity.overall_fidelity)
        tokens = engine.generate([1, 42], max_tokens=10)
    """

    def __init__(self, config: AFEEConfig | None = None) -> None:
        self.cfg = config or AFEEConfig()
        c = self.cfg

        # Sub-systems
        self._core = PolaritonicTensorCore(
            coherence_threshold=c.coherence_threshold,
            noise_scale=c.noise_scale,
            seed=c.seed,
        )

        self._rev = ReversibleInferenceEngine(track_history=False)

        thermo_cfg = ThermoOpticConfig(
            grid_size_nm=(1000, 1000, 500),
            grid_spacing_nm=100,
            dt_ps=10.0,
        )
        self._drift = ThermalDriftLayer(
            thermo_config=thermo_cfg,
            breakdown_temperature_K=c.breakdown_temperature_K,
            reversible_fraction=c.reversible_fraction,
        )

        self._loader = WeightLoader()
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def core(self) -> PolaritonicTensorCore:
        return self._core

    @property
    def drift(self) -> ThermalDriftLayer:
        return self._drift

    @property
    def loader(self) -> WeightLoader:
        return self._loader

    def set_coherence(self, coherence: float) -> None:
        """Update the cosmic-ray coherence level for the tensor core."""
        self._core.set_sector_coherence(coherence)

    def set_availability_map(self, availability: np.ndarray) -> None:
        """Set coherence from Module G availability grid."""
        self._core.set_availability_map(availability)

    def load_weights(
        self, weights: dict[str, np.ndarray] | None = None,
    ) -> LoadReport:
        """Load model weights.

        If *weights* is ``None``, generates random transformer weights
        matching the current config.
        """
        if weights is not None:
            report = self._loader.load_numpy(weights)
        else:
            report = self._loader.generate_random_transformer(
                n_layers=self.cfg.n_layers,
                d_model=self.cfg.d_model,
                d_ff=self.cfg.d_ff,
                n_heads=self.cfg.n_heads,
                vocab_size=self.cfg.vocab_size,
                seed=self.cfg.seed,
            )
        self._loaded = True
        return report

    def load_file(self, path: str) -> LoadReport:
        """Load weights from a file (auto-detect format)."""
        report = self._loader.load_auto(path)
        self._loaded = True
        return report

    def forward(self, token_ids: list[int] | np.ndarray) -> InferenceResult:
        """Run a full forward pass through the transformer.

        Parameters:
            token_ids: Input token IDs (1-D sequence).

        Returns:
            InferenceResult with logits, fidelity, and diagnostics.
        """
        if not self._loaded:
            raise RuntimeError("Call load_weights() before forward()")

        ids = np.asarray(token_ids, dtype=np.int64)
        seq_len = len(ids)
        c = self.cfg
        mode = c.mode
        use_rev = mode == InferenceMode.REVERSIBLE
        use_physics = mode == InferenceMode.PHYSICS_FULL

        # --- Embedding ---
        embed_w = self._loader.get("embed.weight").data
        # Clamp IDs to vocab range
        ids_clamped = np.clip(ids, 0, embed_w.shape[0] - 1)
        x = embed_w[ids_clamped]  # (seq_len, d_model)

        latency_ps = self._loader.get("embed.weight").read_latency_ps

        # --- Transformer layers ---
        for i in range(c.n_layers):
            pre = f"layers.{i}"

            # Attention norm
            norm_g = self._loader.get(f"{pre}.attn_norm").data
            x_norm = self._rms_norm(x, norm_g, use_rev)

            # Self-attention (simplified: no causal mask, no KV cache)
            wq = self._loader.get(f"{pre}.attn.wq").data
            wk = self._loader.get(f"{pre}.attn.wk").data
            wv = self._loader.get(f"{pre}.attn.wv").data
            wo = self._loader.get(f"{pre}.attn.wo").data

            q = self._matmul(x_norm, wq, use_rev, use_physics)
            k = self._matmul(x_norm, wk, use_rev, use_physics)
            v = self._matmul(x_norm, wv, use_rev, use_physics)

            # Scaled dot-product attention
            d_head = c.d_model // c.n_heads
            scale = 1.0 / np.sqrt(d_head)
            scores = self._matmul(q, k.T, use_rev, use_physics) * scale
            attn_weights = self._softmax(scores, use_rev)
            attn_out = self._matmul(attn_weights, v, use_rev, use_physics)
            attn_proj = self._matmul(attn_out, wo, use_rev, use_physics)

            # Residual
            x = self._add(x, attn_proj, use_rev)

            # FFN norm
            ffn_norm_g = self._loader.get(f"{pre}.ffn_norm").data
            x_norm2 = self._rms_norm(x, ffn_norm_g, use_rev)

            # SwiGLU FFN
            w1 = self._loader.get(f"{pre}.ffn.w1").data
            w2 = self._loader.get(f"{pre}.ffn.w2").data
            w3 = self._loader.get(f"{pre}.ffn.w3").data

            gate = self._matmul(x_norm2, w1, use_rev, use_physics)
            up = self._matmul(x_norm2, w3, use_rev, use_physics)
            gate = self._silu(gate, use_rev)
            hidden = gate * up  # element-wise
            if use_physics:
                hidden = self._drift.apply(hidden)
            down = self._matmul(hidden, w2, use_rev, use_physics)

            # Residual
            x = self._add(x, down, use_rev)

            # Accumulate read latency
            for key in [f"{pre}.attn.wq", f"{pre}.attn.wk", f"{pre}.attn.wv",
                        f"{pre}.attn.wo", f"{pre}.ffn.w1", f"{pre}.ffn.w2",
                        f"{pre}.ffn.w3", f"{pre}.attn_norm", f"{pre}.ffn_norm"]:
                latency_ps += self._loader.get(key).read_latency_ps

        # --- Output projection ---
        out_norm_g = self._loader.get("output_norm").data
        x = self._rms_norm(x, out_norm_g, use_rev)
        output_w = self._loader.get("output.weight").data
        logits = self._matmul(x, output_w, use_rev, use_physics)
        latency_ps += self._loader.get("output.weight").read_latency_ps

        # --- Build result ---
        fidelity = self._compute_fidelity()
        return InferenceResult(
            logits=logits,
            fidelity=fidelity,
            load_report=self._loader.report,
            reversibility=self._rev.report,
            thermal_history=self._drift.history,
            latency_ps=latency_ps,
        )

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int = 10,
        temperature: float = 1.0,
    ) -> list[int]:
        """Auto-regressive token generation.

        Returns the full sequence (prompt + generated tokens).
        """
        ids = list(prompt_ids)
        for _ in range(max_tokens):
            result = self.forward(ids)
            # Sample from last position's logits
            last_logits = result.logits[-1]
            if np.any(np.isnan(last_logits)):
                break  # Thermal breakdown — stop generating
            # Temperature-scaled sampling
            probs = _softmax_1d(last_logits / max(temperature, 1e-8))
            probs = np.clip(probs, 0, None)
            prob_sum = probs.sum()
            if prob_sum <= 0 or not np.isfinite(prob_sum):
                break
            probs /= prob_sum
            next_token = int(np.random.choice(len(probs), p=probs))
            ids.append(next_token)
        return ids

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 50,
    ) -> str:
        """Generate degraded text using physics-aware char-level emulation.

        Instead of a full language model, this uses a small static
        vocabulary and coherence-driven character degradation to produce
        readable-but-degraded text that visually demonstrates the effect
        of polariton physics on inference quality.

        At high fidelity (>0.9), the output sentence is mostly readable.
        At low fidelity (<0.3), characters decay into bit-flipped
        substitutions (similar to leetspeak or garbled text).

        Parameters:
            prompt: Input text string.
            max_tokens: Maximum characters to generate.

        Returns:
            The degraded output string.
        """
        emulator = _CharLevelEmulator(seed=self.cfg.seed)

        # Run a forward pass to get the current fidelity state
        prompt_ids = [ord(c) % self.cfg.vocab_size for c in prompt]
        if not prompt_ids:
            prompt_ids = [0]
        result = self.forward(prompt_ids)
        fidelity = result.fidelity.overall_fidelity

        return emulator.generate_degraded(prompt, fidelity, max_tokens)

    def reset_thermal(self) -> None:
        """Cool down the thermal solver (reset to ambient)."""
        self._drift.reset()

    # ------------------------------------------------------------------
    # Internal ops
    # ------------------------------------------------------------------

    def _matmul(
        self, a: np.ndarray, b: np.ndarray,
        use_rev: bool, use_physics: bool = False,
    ) -> np.ndarray:
        if use_rev:
            result = self._rev.matmul(a, b)
        else:
            result = self._core.matmul(a, b)
        if use_physics:
            result = self._drift.apply(result, n_flops=a.shape[0] * b.shape[-1] * (a.shape[-1] if a.ndim > 1 else 1))
        return result

    def _add(
        self, a: np.ndarray, b: np.ndarray, use_rev: bool,
    ) -> np.ndarray:
        if use_rev:
            return self._rev.vector_add(a, b)
        return self._core.vector_add(a, b)

    def _silu(self, x: np.ndarray, use_rev: bool) -> np.ndarray:
        if use_rev:
            return self._rev.silu(x)
        return self._core.silu(x)

    def _softmax(self, x: np.ndarray, use_rev: bool) -> np.ndarray:
        if use_rev:
            return self._rev.softmax(x)
        return self._core.softmax(x)

    def _rms_norm(
        self, x: np.ndarray, gamma: np.ndarray, use_rev: bool,
    ) -> np.ndarray:
        if use_rev:
            rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-5)
            return self._rev.matmul(x / rms, np.diag(gamma)) if gamma.ndim == 1 and x.shape[-1] == gamma.shape[0] else gamma * x / rms
        return self._core.rms_norm(x, gamma)

    def _compute_fidelity(self) -> FidelityReport:
        """Aggregate fidelity from all physics sub-systems."""
        thermal_fid = self._drift.fidelity
        total = self._core.ops_total
        noisy = self._core.ops_noisy
        coherence_fid = 1.0 - (noisy / max(total, 1))
        rev_fid = self._rev.report.fidelity

        overall = thermal_fid * coherence_fid * rev_fid

        return FidelityReport(
            thermal_fidelity=thermal_fid,
            coherence_fidelity=coherence_fid,
            reversibility_fidelity=rev_fid,
            overall_fidelity=overall,
            is_breakdown=self._drift.is_breakdown,
            peak_temperature_K=self._drift.peak_temperature_K,
            phase_error_rad=self._drift.current_phase_error_rad,
        )


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a 1-D array."""
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


# ---------------------------------------------------------------------------
# Char-level emulator for physics-aware text degradation
# ---------------------------------------------------------------------------

# Static vocabulary: common English sentences for demonstration
_STATIC_SENTENCES = [
    "The polariton gate switches at femtosecond speed in sapphire crystal. ",
    "Quantum coherence enables massively parallel optical computation. ",
    "Light pulses propagate through the waveguide lattice carrying data. ",
    "Thermal equilibrium maintains gate fidelity below breakdown temperature. ",
    "Interference patterns encode matrix multiplications in the optical domain. ",
    "Holographic memory stores neural network weights in spin-crossover cells. ",
    "The Aethel chip processes language models at the speed of light. ",
    "Exciton-polariton condensates form bistable switches for logic gates. ",
]

# Bit-flip substitution map: visually similar characters
_BIT_FLIP_MAP: dict[str, str] = {
    "a": "4", "A": "4", "e": "3", "E": "3",
    "i": "1", "I": "!", "o": "0", "O": "0",
    "s": "5", "S": "$", "t": "7", "T": "7",
    "l": "|", "g": "9", "b": "6", "B": "8",
    "z": "2", "Z": "2", "n": "^", "r": "®",
    "c": "(", "d": ")", "h": "#", "u": "µ",
    "m": "^^", "w": "vv", "p": "¶", "q": "9",
    "f": "ƒ", "v": "√", "x": "×", "y": "¥",
    "k": "κ", "j": "]",
}


class _CharLevelEmulator:
    """Physics-aware character-level text emulator.

    Uses the overall fidelity score to determine how much each character
    in a generated sentence is degraded:

    * **fidelity >= 0.9** — text is fully readable, occasional typo.
    * **fidelity ~0.5** — noticeable corruption, some leetspeak-style subs.
    * **fidelity <= 0.3** — heavy corruption, mostly garbled/unreadable.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def generate_degraded(
        self,
        prompt: str,
        fidelity: float,
        max_chars: int = 200,
    ) -> str:
        """Generate a text response and degrade it based on fidelity.

        Parameters:
            prompt: Input text (used to seed sentence selection).
            fidelity: Overall inference fidelity [0, 1].
            max_chars: Maximum output length.

        Returns:
            Degraded text string.
        """
        fidelity = max(0.0, min(1.0, fidelity))

        # Select sentences based on prompt hash
        seed_val = sum(ord(c) for c in prompt) if prompt else 0
        rng = np.random.default_rng(seed_val)
        indices = rng.permutation(len(_STATIC_SENTENCES))

        # Build clean output from static vocabulary
        clean = ""
        for idx in indices:
            clean += _STATIC_SENTENCES[idx]
            if len(clean) >= max_chars:
                break
        # Repeat if needed
        while len(clean) < max_chars:
            clean += _STATIC_SENTENCES[int(rng.integers(len(_STATIC_SENTENCES)))]
        clean = clean[:max_chars]

        # Apply physics-aware degradation
        # Corruption probability per character = 1 - fidelity
        corruption_prob = 1.0 - fidelity

        result = []
        for ch in clean:
            if self._rng.random() < corruption_prob:
                result.append(self._degrade_char(ch))
            else:
                result.append(ch)

        return "".join(result)

    def _degrade_char(self, ch: str) -> str:
        """Degrade a single character using bit-flip simulation.

        Preference order:
        1. Visually similar substitution (leetspeak-style).
        2. Case flip.
        3. Random printable ASCII character.
        """
        roll = self._rng.random()

        if roll < 0.5 and ch in _BIT_FLIP_MAP:
            # Visually similar substitution
            return _BIT_FLIP_MAP[ch]
        elif roll < 0.7 and ch.isalpha():
            # Case flip
            return ch.swapcase()
        elif roll < 0.85:
            # Nearby ASCII character (±1..3)
            offset = int(self._rng.integers(-3, 4))
            new_ord = ord(ch) + offset
            if 32 <= new_ord <= 126:
                return chr(new_ord)
            return ch
        else:
            # Random printable ASCII
            return chr(int(self._rng.integers(33, 127)))
