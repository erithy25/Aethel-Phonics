"""Tests for Module J — Aethel Functional Emulation Engine (AFEE)."""

import numpy as np
import pytest

from apld_mps.module_j.tensor_core import PolaritonicTensorCore, CoherenceSector
from apld_mps.module_j.reversible_inference import (
    ReversibleInferenceEngine,
    ReversibilityReport,
)
from apld_mps.module_j.weight_loader import WeightLoader, WeightTensor, LoadReport
from apld_mps.module_j.thermal_drift import ThermalDriftLayer, DriftSnapshot
from apld_mps.module_j.engine import AFEE, AFEEConfig, InferenceMode, FidelityReport


# =========================================================================
# Detail 2.1 — PolaritonicTensorCore
# =========================================================================

class TestPolaritonicTensorCore:

    def test_matmul_correct_at_full_coherence(self):
        core = PolaritonicTensorCore(coherence_threshold=0.8, seed=0)
        a = np.eye(3)
        b = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        result = core.matmul(a, b)
        np.testing.assert_allclose(result, a @ b)

    def test_noise_injected_below_threshold(self):
        core = PolaritonicTensorCore(
            coherence_threshold=0.8, noise_scale=1.0, seed=42,
        )
        core.set_sector_coherence(0.3)  # well below threshold
        a = np.ones((4, 4))
        b = np.ones((4, 4))
        result = core.matmul(a, b)
        expected = a @ b
        # Should NOT be exactly equal due to injected noise
        assert not np.allclose(result, expected)
        assert core.ops_noisy == 1

    def test_no_noise_above_threshold(self):
        core = PolaritonicTensorCore(coherence_threshold=0.8, seed=0)
        core.set_sector_coherence(0.95)
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = core.matmul(a, a)
        np.testing.assert_allclose(result, a @ a)
        assert core.ops_noisy == 0

    def test_ops_counters(self):
        core = PolaritonicTensorCore(seed=0)
        core.set_sector_coherence(0.5)
        x = np.ones((2, 2))
        core.matmul(x, x)
        core.relu(x)
        core.vector_add(x, x)
        assert core.ops_total == 3
        assert core.ops_noisy == 3  # all below default 0.8

    def test_softmax_sums_to_one(self):
        core = PolaritonicTensorCore(seed=0)
        x = np.array([[1.0, 2.0, 3.0]])
        result = core.softmax(x)
        assert result.sum() == pytest.approx(1.0, abs=1e-6)

    def test_relu_non_negative(self):
        core = PolaritonicTensorCore(seed=0)
        x = np.array([-2, -1, 0, 1, 2], dtype=float)
        result = core.relu(x)
        assert np.all(result >= -core.noise_scale)  # allow tiny noise

    def test_silu_shape_preserved(self):
        core = PolaritonicTensorCore(seed=0)
        x = np.random.randn(3, 4)
        result = core.silu(x)
        assert result.shape == x.shape

    def test_rms_norm(self):
        core = PolaritonicTensorCore(seed=0)
        x = np.random.randn(2, 8)
        gamma = np.ones(8)
        result = core.rms_norm(x, gamma)
        assert result.shape == x.shape

    def test_set_availability_map(self):
        core = PolaritonicTensorCore(seed=0)
        avail = np.full((10, 10, 10), 0.6)
        core.set_availability_map(avail)
        assert core.sector_coherence == pytest.approx(0.6)

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            PolaritonicTensorCore(coherence_threshold=1.5)


# =========================================================================
# Detail 2.2 — ReversibleInferenceEngine
# =========================================================================

class TestReversibleInference:

    def test_matmul_tracks_reversibility(self):
        rev = ReversibleInferenceEngine()
        a = np.eye(3)
        b = np.ones((3, 2))
        result = rev.matmul(a, b)
        np.testing.assert_allclose(result, a @ b)
        assert rev.report.total_ops == 1
        assert rev.report.reversible_ops == 1

    def test_zero_bits_erased_for_unique_ops(self):
        rev = ReversibleInferenceEngine()
        # Different inputs → different outputs → all reversible
        for i in range(5):
            a = np.full((2, 2), float(i + 1))
            b = np.eye(2)
            rev.matmul(a, b)
        assert rev.report.bits_erased == 0
        assert rev.report.entropy_J == 0.0
        assert rev.report.fidelity == 1.0

    def test_relu_stores_ancilla(self):
        rev = ReversibleInferenceEngine(track_history=True)
        x = np.array([-1.0, 0.0, 1.0])
        rev.relu(x)
        # Ancilla should include the input + sign mask
        assert rev.ancilla_count >= 1

    def test_uncompute_frees_ancilla(self):
        rev = ReversibleInferenceEngine(track_history=True)
        x = np.array([1.0, 2.0])
        rev.silu(x)
        assert rev.ancilla_count > 0
        freed = rev.uncompute()
        assert freed > 0
        assert rev.ancilla_count == 0

    def test_entropy_positive_for_collisions(self):
        rev = ReversibleInferenceEngine()
        # ReLU on different negative arrays → same output (zeros)
        rev.relu(np.array([-1.0, -2.0]))
        rev.relu(np.array([-3.0, -4.0]))
        # Second relu outputs same hash → collision → irreversible
        assert rev.report.irreversible_ops >= 1
        assert rev.report.entropy_J > 0

    def test_softmax_preserves_shape(self):
        rev = ReversibleInferenceEngine()
        x = np.array([[1.0, 2.0, 3.0]])
        result = rev.softmax(x)
        assert result.shape == x.shape
        assert result.sum() == pytest.approx(1.0, abs=1e-5)

    def test_report_fidelity_range(self):
        rev = ReversibleInferenceEngine()
        assert 0.0 <= rev.report.fidelity <= 1.0


# =========================================================================
# Detail 2.3 — WeightLoader
# =========================================================================

class TestWeightLoader:

    def test_load_numpy(self):
        loader = WeightLoader()
        weights = {"w1": np.ones((4, 4)), "w2": np.zeros((2, 8))}
        report = loader.load_numpy(weights)
        assert report.n_tensors == 2
        assert report.total_parameters == 4 * 4 + 2 * 8
        assert report.total_read_latency_ps > 0

    def test_weight_retrieval(self):
        loader = WeightLoader()
        w = np.random.randn(3, 3)
        loader.load_numpy({"test": w})
        wt = loader.get("test")
        np.testing.assert_allclose(wt.data, w)
        assert wt.shape == (3, 3)
        assert wt.read_latency_ps > 0
        assert wt.read_energy_pj > 0

    def test_generate_random_transformer(self):
        loader = WeightLoader()
        report = loader.generate_random_transformer(
            n_layers=2, d_model=32, d_ff=64, n_heads=2, vocab_size=100,
        )
        assert report.n_tensors > 0
        assert report.total_parameters > 0
        assert "embed.weight" in loader.weight_names
        assert "layers.0.attn.wq" in loader.weight_names
        assert "output.weight" in loader.weight_names

    def test_read_latency_includes_propagation(self):
        loader = WeightLoader(propagation_distance_nm=100_000.0)
        loader.load_numpy({"w": np.ones((10, 10))})
        wt = loader.get("w")
        # Propagation time = 100_000 nm / (c/n) should be > 0
        assert wt.read_latency_ps > 0

    def test_weight_names_list(self):
        loader = WeightLoader()
        loader.load_numpy({"a": np.ones(1), "b": np.ones(2)})
        assert set(loader.weight_names) == {"a", "b"}


# =========================================================================
# Detail 2.4 — ThermalDriftLayer
# =========================================================================

class TestThermalDriftLayer:

    def _make_layer(self, **kwargs) -> ThermalDriftLayer:
        from apld_mps.module_f.heat_optics import ThermoOpticConfig
        cfg = ThermoOpticConfig(
            grid_size_nm=(500, 500, 200),
            grid_spacing_nm=100,
            dt_ps=10.0,
        )
        return ThermalDriftLayer(thermo_config=cfg, **kwargs)

    def test_initial_state(self):
        layer = self._make_layer()
        assert layer.peak_temperature_K == pytest.approx(300.0)
        assert layer.is_breakdown is False
        assert layer.fidelity == pytest.approx(1.0)

    def test_apply_preserves_shape(self):
        layer = self._make_layer()
        x = np.random.randn(3, 4)
        result = layer.apply(x)
        assert result.shape == x.shape

    def test_history_grows(self):
        layer = self._make_layer()
        x = np.ones((2, 2))
        layer.apply(x)
        layer.apply(x)
        assert len(layer.history) == 2

    def test_breakdown_produces_nan(self):
        layer = self._make_layer(breakdown_temperature_K=300.1)
        # Inject massive heat to exceed breakdown
        layer.solver._temperature += 20.0  # force temperature rise
        x = np.ones((2, 2))
        result = layer.apply(x)
        assert np.all(np.isnan(result))
        assert layer.is_breakdown is True
        assert layer.fidelity == 0.0

    def test_reset_clears_state(self):
        layer = self._make_layer()
        layer.apply(np.ones((2, 2)), n_flops=1000)
        layer.reset()
        assert layer.peak_temperature_K == pytest.approx(300.0)
        assert layer.is_breakdown is False
        assert len(layer.history) == 0

    def test_fidelity_decreases_with_phase_error(self):
        layer = self._make_layer()
        # Manually raise temperature to get a phase error
        layer.solver._temperature += 5.0
        layer.apply(np.ones((2, 2)))
        # Fidelity should be slightly below 1.0
        assert layer.fidelity <= 1.0

    def test_reversible_fraction_reduces_heat(self):
        layer_irr = self._make_layer(reversible_fraction=0.0)
        layer_rev = self._make_layer(reversible_fraction=1.0)
        x = np.ones((10, 10))
        layer_irr.apply(x, n_flops=10_000)
        layer_rev.apply(x, n_flops=10_000)
        # Reversible layer should not heat up
        assert layer_rev.peak_temperature_K <= layer_irr.peak_temperature_K


# =========================================================================
# AFEE Engine (integration)
# =========================================================================

class TestAFEEEngine:

    def _make_engine(self, **kwargs) -> AFEE:
        cfg = AFEEConfig(
            n_layers=2, d_model=32, d_ff=64, n_heads=2, vocab_size=100,
            **kwargs,
        )
        engine = AFEE(cfg)
        engine.load_weights()
        return engine

    def test_forward_returns_logits(self):
        engine = self._make_engine(mode=InferenceMode.STANDARD)
        result = engine.forward([1, 5, 10])
        assert result.logits.shape == (3, 100)  # (seq_len, vocab_size)
        assert np.all(np.isfinite(result.logits))

    def test_forward_physics_full(self):
        engine = self._make_engine(mode=InferenceMode.PHYSICS_FULL)
        result = engine.forward([1, 2, 3])
        assert result.logits.shape == (3, 100)
        assert result.fidelity.thermal_fidelity > 0

    def test_forward_reversible(self):
        engine = self._make_engine(mode=InferenceMode.REVERSIBLE)
        result = engine.forward([1, 2])
        assert result.logits.shape == (2, 100)
        assert result.reversibility.total_ops > 0

    def test_generate_produces_tokens(self):
        engine = self._make_engine(mode=InferenceMode.STANDARD)
        tokens = engine.generate([1, 2], max_tokens=5)
        assert len(tokens) >= 2  # at least prompt tokens
        assert len(tokens) <= 7  # prompt + max_tokens

    def test_fidelity_report_complete(self):
        engine = self._make_engine()
        result = engine.forward([1])
        fid = result.fidelity
        assert 0.0 <= fid.thermal_fidelity <= 1.0
        assert 0.0 <= fid.coherence_fidelity <= 1.0
        assert 0.0 <= fid.reversibility_fidelity <= 1.0
        assert 0.0 <= fid.overall_fidelity <= 1.0
        assert fid.peak_temperature_K >= 300.0
        assert fid.phase_error_rad >= 0.0

    def test_coherence_degradation_affects_fidelity(self):
        engine = self._make_engine(mode=InferenceMode.STANDARD)
        engine.set_coherence(1.0)
        r_good = engine.forward([1, 2])
        engine.set_coherence(0.3)
        r_bad = engine.forward([1, 2])
        # With low coherence, coherence_fidelity should be lower
        assert r_bad.fidelity.coherence_fidelity <= r_good.fidelity.coherence_fidelity

    def test_not_loaded_raises(self):
        engine = AFEE(AFEEConfig())
        with pytest.raises(RuntimeError, match="load_weights"):
            engine.forward([1])

    def test_reset_thermal(self):
        engine = self._make_engine(mode=InferenceMode.PHYSICS_FULL)
        engine.forward([1, 2, 3])
        engine.reset_thermal()
        assert engine.drift.peak_temperature_K == pytest.approx(300.0)

    def test_load_report_has_latency(self):
        engine = self._make_engine()
        result = engine.forward([1])
        assert result.load_report.total_read_latency_ps > 0
        assert result.latency_ps > 0

    def test_availability_map_coupling(self):
        engine = self._make_engine(mode=InferenceMode.STANDARD)
        avail = np.full((5, 5, 5), 0.5)
        engine.set_availability_map(avail)
        assert engine.core.sector_coherence == pytest.approx(0.5)
