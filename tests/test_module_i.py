"""Tests for module_i — AI Tensor Compiler."""

import pytest

from apld_mps.module_i import (
    OpType,
    TensorShape,
    ComputeNode,
    ComputeGraph,
    ModelParser,
    ClusterType,
    GateCluster,
    PolaritonicMapping,
    PolaritonicMapper,
    SCOCellSpec,
    WeightBlock,
    MemoryRegion,
    ReadPulseResult,
    HolographicMemoryInterface,
    ResourceEstimate,
    ResourceEstimator,
    LayerLatency,
    LatencyReport,
    LatencyAnalyser,
)


# ========================================================================
# Model Parser
# ========================================================================

class TestTensorShape:
    def test_numel(self):
        s = TensorShape(dims=(4, 8, 16))
        assert s.numel == 512

    def test_ndim(self):
        s = TensorShape(dims=(2, 3))
        assert s.ndim == 2

    def test_single_element(self):
        s = TensorShape(dims=(1,))
        assert s.numel == 1


class TestComputeNode:
    def test_matmul_flops(self):
        node = ComputeNode(
            node_id="mm0",
            op_type=OpType.MATMUL,
            input_shapes=[
                TensorShape(dims=(16, 64)),
                TensorShape(dims=(64, 32)),
            ],
            output_shape=TensorShape(dims=(16, 32)),
        )
        flops = node.estimate_flops()
        # 2 * M * K * N = 2 * 16 * 64 * 32 = 65536
        assert flops == 65536

    def test_relu_flops(self):
        node = ComputeNode(
            node_id="relu0",
            op_type=OpType.RELU,
            output_shape=TensorShape(dims=(16, 32)),
        )
        flops = node.estimate_flops()
        assert flops == 512

    def test_softmax_flops(self):
        node = ComputeNode(
            node_id="sm0",
            op_type=OpType.SOFTMAX,
            output_shape=TensorShape(dims=(8, 64, 64)),
        )
        flops = node.estimate_flops()
        assert flops == 5 * 8 * 64 * 64


class TestOpType:
    def test_from_onnx_matmul(self):
        assert OpType.from_onnx_op("MatMul") == OpType.MATMUL

    def test_from_onnx_gemm(self):
        assert OpType.from_onnx_op("Gemm") == OpType.MATMUL

    def test_from_onnx_unknown(self):
        assert OpType.from_onnx_op("CustomOp") == OpType.UNKNOWN


class TestComputeGraph:
    def test_total_flops(self):
        n1 = ComputeNode(node_id="a", op_type=OpType.RELU,
                         output_shape=TensorShape(dims=(100,)))
        n1.estimate_flops()
        n2 = ComputeNode(node_id="b", op_type=OpType.RELU,
                         output_shape=TensorShape(dims=(200,)))
        n2.estimate_flops()
        g = ComputeGraph(nodes=[n1, n2])
        assert g.total_flops == 300

    def test_topological_order(self):
        from apld_mps.module_i.model_parser import ComputeEdge
        n1 = ComputeNode(node_id="a", op_type=OpType.RELU)
        n2 = ComputeNode(node_id="b", op_type=OpType.RELU)
        n3 = ComputeNode(node_id="c", op_type=OpType.ADD)
        edges = [ComputeEdge(src_id="a", dst_id="c"),
                 ComputeEdge(src_id="b", dst_id="c")]
        g = ComputeGraph(nodes=[n1, n2, n3], edges=edges)
        order = g.topological_order()
        ids = [n.node_id for n in order]
        assert ids.index("a") < ids.index("c")
        assert ids.index("b") < ids.index("c")


class TestModelParser:
    def test_transformer_spec_basic(self):
        parser = ModelParser()
        graph = parser.from_transformer_spec(
            name="tiny",
            n_layers=2,
            d_model=64,
            n_heads=4,
            vocab_size=1000,
            seq_len=32,
        )
        assert graph.name == "tiny"
        assert graph.total_parameters > 0
        assert graph.node_count > 0
        assert graph.total_flops > 0

    def test_transformer_spec_7b(self):
        parser = ModelParser()
        graph = parser.from_transformer_spec(
            name="llama-7b",
            n_layers=32,
            d_model=4096,
            n_heads=32,
            vocab_size=32000,
            seq_len=2048,
        )
        # ~6.7B parameters expected
        assert graph.total_parameters > 6_000_000_000
        assert graph.total_parameters < 8_000_000_000


# ========================================================================
# Polaritonic Mapping
# ========================================================================

class TestPolaritonicMapper:
    def test_matmul_maps_to_interference_grid(self):
        node = ComputeNode(
            node_id="mm",
            op_type=OpType.MATMUL,
            input_shapes=[
                TensorShape(dims=(8, 16)),
                TensorShape(dims=(16, 4)),
            ],
            output_shape=TensorShape(dims=(8, 4)),
        )
        node.estimate_flops()
        graph = ComputeGraph(nodes=[node])
        mapper = PolaritonicMapper()
        mapping = mapper.map(graph)
        assert len(mapping.clusters) == 1
        assert mapping.clusters[0].cluster_type == ClusterType.INTERFERENCE_GRID
        assert mapping.clusters[0].gate_count > 0

    def test_relu_maps_to_threshold(self):
        node = ComputeNode(
            node_id="relu",
            op_type=OpType.RELU,
            output_shape=TensorShape(dims=(16,)),
        )
        graph = ComputeGraph(nodes=[node])
        mapper = PolaritonicMapper()
        mapping = mapper.map(graph)
        assert mapping.clusters[0].cluster_type == ClusterType.THRESHOLD_NONLINEAR
        assert mapping.clusters[0].gate_count == 16

    def test_softmax_maps_to_normalisation(self):
        node = ComputeNode(
            node_id="sm",
            op_type=OpType.SOFTMAX,
            output_shape=TensorShape(dims=(8, 8)),
        )
        graph = ComputeGraph(nodes=[node])
        mapper = PolaritonicMapper()
        mapping = mapper.map(graph)
        assert mapping.clusters[0].cluster_type == ClusterType.NORMALISATION_CASCADE
        # 3 gates per element for normalisation
        assert mapping.clusters[0].gate_count == 64 * 3

    def test_total_gate_count(self):
        n1 = ComputeNode(node_id="r1", op_type=OpType.RELU,
                         output_shape=TensorShape(dims=(10,)))
        n2 = ComputeNode(node_id="r2", op_type=OpType.RELU,
                         output_shape=TensorShape(dims=(20,)))
        graph = ComputeGraph(nodes=[n1, n2])
        mapping = PolaritonicMapper().map(graph)
        assert mapping.total_gate_count == 30

    def test_total_power(self):
        node = ComputeNode(node_id="r", op_type=OpType.RELU,
                           output_shape=TensorShape(dims=(100,)))
        mapping = PolaritonicMapper().map(ComputeGraph(nodes=[node]))
        assert mapping.total_power_mw == pytest.approx(100 * 0.01)


# ========================================================================
# Memory Interface
# ========================================================================

class TestSCOCellSpec:
    def test_defaults(self):
        spec = SCOCellSpec()
        assert spec.cell_pitch_nm == 50.0
        assert spec.read_time_ps == 0.1


class TestWeightBlock:
    def test_total_weights(self):
        wb = WeightBlock(block_id="w0", rows=4, cols=8)
        assert wb.total_weights == 32
        assert wb.total_bits == 256
        assert wb.size_bytes == 32


class TestMemoryRegion:
    def test_capacity(self):
        region = MemoryRegion(
            region_id="r0",
            volume_nm=(1000.0, 1000.0, 1000.0),
            cell_spec=SCOCellSpec(cell_pitch_nm=100.0),
        )
        # 10 * 10 * 10 = 1000 cells
        assert region.capacity_cells == 1000
        assert region.capacity_bits == 1000

    def test_utilisation(self):
        region = MemoryRegion(
            region_id="r0",
            volume_nm=(1000.0, 1000.0, 1000.0),
            cell_spec=SCOCellSpec(cell_pitch_nm=100.0),
        )
        region.weight_blocks.append(
            WeightBlock(block_id="w", rows=1, cols=100, bits_per_weight=1)
        )
        assert region.utilisation == pytest.approx(0.1)


class TestHolographicMemoryInterface:
    def test_allocate_and_store(self):
        mem = HolographicMemoryInterface()
        region = mem.allocate_region("r0", (10_000.0, 10_000.0, 10_000.0))
        wb = mem.store_weights(region, "w0", rows=4, cols=4)
        assert wb.total_weights == 16
        assert len(region.weight_blocks) == 1

    def test_store_overflow(self):
        mem = HolographicMemoryInterface(
            cell_spec=SCOCellSpec(cell_pitch_nm=100.0)
        )
        region = mem.allocate_region("r0", (100.0, 100.0, 100.0))
        # 1 cell, 1 bit capacity — trying to store 8 bits should fail
        with pytest.raises(ValueError, match="capacity exceeded"):
            mem.store_weights(region, "big", rows=1, cols=2, bits_per_weight=8)

    def test_read_weights(self):
        mem = HolographicMemoryInterface()
        region = mem.allocate_region("r0", (10_000.0, 10_000.0, 10_000.0))
        mem.store_weights(region, "w0", rows=4, cols=4)
        result = mem.read_weights(region, "w0", propagation_distance_nm=100_000.0)
        assert result.read_time_ps == 0.1
        assert result.propagation_time_ps > 0
        assert result.total_latency_ps > result.read_time_ps

    def test_read_missing_block(self):
        mem = HolographicMemoryInterface()
        region = mem.allocate_region("r0")
        with pytest.raises(KeyError):
            mem.read_weights(region, "nonexistent")

    def test_allocate_model_weights(self):
        mem = HolographicMemoryInterface()
        regions = mem.allocate_model_weights(
            total_parameters=1_000_000,
            bits_per_weight=8,
            region_volume_nm=(1_000_000.0, 1_000_000.0, 100_000.0),
        )
        assert len(regions) >= 1
        total_stored = sum(
            wb.total_weights
            for r in regions
            for wb in r.weight_blocks
        )
        assert total_stored == 1_000_000


# ========================================================================
# Resource Estimator
# ========================================================================

class TestResourceEstimator:
    def test_basic_estimate(self):
        parser = ModelParser()
        graph = parser.from_transformer_spec(
            name="tiny",
            n_layers=2,
            d_model=64,
            n_heads=4,
            vocab_size=1000,
            seq_len=32,
        )
        est = ResourceEstimator()
        result = est.estimate(graph)
        assert result.total_gate_count > 0
        assert result.layer_count >= 1
        assert result.monolith_volume_nm[0] > 0
        assert result.memory_regions >= 1
        assert result.total_power_w > 0

    def test_estimate_from_spec(self):
        est = ResourceEstimator()
        result = est.estimate_from_spec(
            name="small",
            n_layers=4,
            d_model=128,
            n_heads=4,
            vocab_size=2000,
            seq_len=64,
        )
        assert result.model_name == "small"
        assert result.total_parameters > 0
        assert result.weight_storage_bytes > 0

    def test_summary_format(self):
        est = ResourceEstimator()
        result = est.estimate_from_spec(
            name="test-model",
            n_layers=1,
            d_model=32,
            n_heads=2,
            vocab_size=100,
            seq_len=8,
        )
        summary = result.summary()
        assert "test-model" in summary
        assert "Polariton gates" in summary
        assert "Z-layers" in summary

    def test_7b_estimate(self):
        est = ResourceEstimator()
        result = est.estimate_from_spec(
            name="llama-7b",
            n_layers=32,
            d_model=4096,
            n_heads=32,
            vocab_size=32000,
            seq_len=2048,
        )
        assert result.total_parameters > 6_000_000_000
        assert result.total_gate_count > result.total_parameters
        # At 8-bit: ~7 GB of weight storage
        assert result.weight_storage_bytes > 6_000_000_000


# ========================================================================
# Latency Analyser
# ========================================================================

class TestLatencyAnalyser:
    def test_basic_analysis(self):
        analyser = LatencyAnalyser()
        report = analyser.analyse_from_spec(
            name="tiny",
            n_layers=2,
            d_model=64,
            n_heads=4,
            vocab_size=1000,
            seq_len=32,
        )
        assert report.model_name == "tiny"
        assert len(report.layer_latencies) == 2
        assert report.total_latency_ps > 0
        assert report.tokens_per_second > 0

    def test_speed_of_light(self):
        analyser = LatencyAnalyser(n_sapphire=1.77)
        from apld_mps.constants import C_LIGHT
        expected = C_LIGHT / 1.77
        report = analyser.analyse_from_spec(
            name="test", n_layers=1, d_model=32, n_heads=2,
            vocab_size=100, seq_len=8,
        )
        assert report.speed_of_light_in_medium_m_s == pytest.approx(expected)

    def test_more_layers_more_latency(self):
        analyser = LatencyAnalyser()
        r1 = analyser.analyse_from_spec(
            name="shallow", n_layers=2, d_model=64, n_heads=4,
            vocab_size=1000, seq_len=32,
        )
        r2 = analyser.analyse_from_spec(
            name="deep", n_layers=8, d_model=64, n_heads=4,
            vocab_size=1000, seq_len=32,
        )
        assert r2.total_latency_ps > r1.total_latency_ps
        assert r1.tokens_per_second > r2.tokens_per_second

    def test_summary_format(self):
        analyser = LatencyAnalyser()
        report = analyser.analyse_from_spec(
            name="test-model", n_layers=2, d_model=64, n_heads=4,
            vocab_size=1000, seq_len=32,
        )
        summary = report.summary()
        assert "test-model" in summary
        assert "Tokens per second" in summary
        assert "Light speed" in summary

    def test_layer_latency_breakdown(self):
        analyser = LatencyAnalyser()
        report = analyser.analyse_from_spec(
            name="breakdown", n_layers=4, d_model=128, n_heads=4,
            vocab_size=2000, seq_len=64,
        )
        for ll in report.layer_latencies:
            assert ll.total_ps > 0
            assert ll.attention_ps >= 0
            assert ll.ffn_ps >= 0
            assert ll.memory_read_ps > 0
            assert ll.propagation_ps > 0

    def test_parse_layer_index(self):
        assert LatencyAnalyser._parse_layer_index("layer5_qkv_42") == 5
        assert LatencyAnalyser._parse_layer_index("layer0_ln1_0") == 0
        assert LatencyAnalyser._parse_layer_index("embed_0") is None
