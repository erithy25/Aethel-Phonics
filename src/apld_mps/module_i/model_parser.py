"""Model Parser — extract computation graphs from ONNX / PyTorch models.

Parses neural-network graphs (ONNX protobuf or ``torch.nn.Module`` objects)
into a hardware-agnostic intermediate representation that the polaritonic
mapping stage can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Sequence


class OpType(Enum):
    """Supported tensor operations."""

    MATMUL = auto()
    CONV2D = auto()
    RELU = auto()
    GELU = auto()
    SOFTMAX = auto()
    LAYERNORM = auto()
    ADD = auto()
    MUL = auto()
    TRANSPOSE = auto()
    RESHAPE = auto()
    EMBEDDING = auto()
    UNKNOWN = auto()

    @classmethod
    def from_onnx_op(cls, op_type: str) -> OpType:
        _MAP = {
            "MatMul": cls.MATMUL,
            "Gemm": cls.MATMUL,
            "Conv": cls.CONV2D,
            "Relu": cls.RELU,
            "Gelu": cls.GELU,
            "Softmax": cls.SOFTMAX,
            "LayerNormalization": cls.LAYERNORM,
            "Add": cls.ADD,
            "Mul": cls.MUL,
            "Transpose": cls.TRANSPOSE,
            "Reshape": cls.RESHAPE,
            "Gather": cls.EMBEDDING,
        }
        return _MAP.get(op_type, cls.UNKNOWN)


@dataclass
class TensorShape:
    """Shape descriptor for a tensor flowing between operations."""

    dims: tuple[int, ...]

    @property
    def numel(self) -> int:
        result = 1
        for d in self.dims:
            if d > 0:
                result *= d
        return result

    @property
    def ndim(self) -> int:
        return len(self.dims)


@dataclass
class ComputeNode:
    """A single operation in the computation graph.

    Attributes:
        node_id: Unique identifier within the graph.
        op_type: The tensor operation performed.
        input_shapes: Shapes of each input tensor.
        output_shape: Shape of the output tensor.
        attributes: Extra operation parameters (e.g. axis for Softmax).
        flops: Estimated floating-point operations for this node.
    """

    node_id: str
    op_type: OpType
    input_shapes: list[TensorShape] = field(default_factory=list)
    output_shape: TensorShape = field(default_factory=lambda: TensorShape(dims=(1,)))
    attributes: dict[str, Any] = field(default_factory=dict)
    flops: int = 0

    def estimate_flops(self) -> int:
        """Estimate FLOPs for this node based on shapes and operation type."""
        if self.op_type == OpType.MATMUL and len(self.input_shapes) >= 2:
            a, b = self.input_shapes[0], self.input_shapes[1]
            # (... , M, K) x (... , K, N) → 2*M*K*N
            m = a.dims[-2] if a.ndim >= 2 else 1
            k = a.dims[-1]
            n = b.dims[-1] if b.ndim >= 2 else 1
            batch = 1
            for d in a.dims[:-2]:
                if d > 0:
                    batch *= d
            self.flops = batch * 2 * m * k * n
        elif self.op_type == OpType.CONV2D and len(self.input_shapes) >= 2:
            inp, kernel = self.input_shapes[0], self.input_shapes[1]
            # Simple estimate: 2 * Cout * Cin * Kh * Kw * Hout * Wout
            if kernel.ndim >= 4 and inp.ndim >= 4:
                cout, cin, kh, kw = kernel.dims[:4]
                hout = inp.dims[2]
                wout = inp.dims[3]
                self.flops = 2 * cout * cin * kh * kw * hout * wout
        elif self.op_type in (OpType.RELU, OpType.GELU):
            self.flops = self.output_shape.numel
        elif self.op_type == OpType.SOFTMAX:
            # 5n per row (max, sub, exp, sum, div)
            self.flops = 5 * self.output_shape.numel
        elif self.op_type == OpType.LAYERNORM:
            self.flops = 5 * self.output_shape.numel
        elif self.op_type in (OpType.ADD, OpType.MUL):
            self.flops = self.output_shape.numel
        else:
            self.flops = 0
        return self.flops


@dataclass
class ComputeEdge:
    """Directed edge in the computation graph (data dependency)."""

    src_id: str
    dst_id: str
    tensor_shape: TensorShape = field(default_factory=lambda: TensorShape(dims=(1,)))


@dataclass
class ComputeGraph:
    """Hardware-agnostic intermediate representation of a neural network.

    Attributes:
        name: Model name.
        nodes: Ordered list of compute nodes.
        edges: Data-dependency edges.
        total_parameters: Total number of trainable parameters.
    """

    name: str = "unnamed"
    nodes: list[ComputeNode] = field(default_factory=list)
    edges: list[ComputeEdge] = field(default_factory=list)
    total_parameters: int = 0

    @property
    def total_flops(self) -> int:
        return sum(n.flops for n in self.nodes)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def topological_order(self) -> list[ComputeNode]:
        """Return nodes in topological (dependency-respecting) order."""
        adj: dict[str, list[str]] = {n.node_id: [] for n in self.nodes}
        in_deg: dict[str, int] = {n.node_id: 0 for n in self.nodes}
        for e in self.edges:
            if e.src_id in adj and e.dst_id in in_deg:
                adj[e.src_id].append(e.dst_id)
                in_deg[e.dst_id] += 1

        queue = [nid for nid, d in in_deg.items() if d == 0]
        order: list[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for child in adj.get(nid, []):
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)

        id_to_node = {n.node_id: n for n in self.nodes}
        return [id_to_node[nid] for nid in order if nid in id_to_node]


class ModelParser:
    """Parse neural-network models into :class:`ComputeGraph`.

    Supports two input formats:
    * **ONNX protobuf** — via :meth:`parse_onnx`.
    * **Transformer spec** — a lightweight dictionary describing a standard
      Transformer architecture via :meth:`from_transformer_spec`.
    """

    def parse_onnx(self, model: Any) -> ComputeGraph:
        """Parse an ONNX ``ModelProto`` into a :class:`ComputeGraph`.

        Parameters:
            model: An ``onnx.ModelProto`` object.

        Returns:
            Populated :class:`ComputeGraph`.
        """
        graph = ComputeGraph(name=getattr(model, "doc_string", "onnx-model"))
        onnx_graph = model.graph

        # Count parameters from initialisers
        total_params = 0
        for init in onnx_graph.initializer:
            n = 1
            for d in init.dims:
                n *= d
            total_params += n
        graph.total_parameters = total_params

        # Build shape map from value_info + initializer
        shape_map: dict[str, TensorShape] = {}
        for vi in list(onnx_graph.input) + list(onnx_graph.output) + list(onnx_graph.value_info):
            tt = vi.type.tensor_type
            if tt.HasField("shape"):
                dims = []
                for d in tt.shape.dim:
                    dims.append(d.dim_value if d.dim_value > 0 else 1)
                shape_map[vi.name] = TensorShape(dims=tuple(dims))

        # Parse nodes
        for i, node in enumerate(onnx_graph.node):
            op = OpType.from_onnx_op(node.op_type)
            in_shapes = [shape_map.get(inp, TensorShape(dims=(1,))) for inp in node.input if inp]
            out_name = node.output[0] if node.output else f"out_{i}"
            out_shape = shape_map.get(out_name, TensorShape(dims=(1,)))

            cn = ComputeNode(
                node_id=node.name or f"node_{i}",
                op_type=op,
                input_shapes=in_shapes,
                output_shape=out_shape,
            )
            cn.estimate_flops()
            graph.nodes.append(cn)

            # Edges from inputs
            for inp in node.input:
                if inp:
                    graph.edges.append(ComputeEdge(
                        src_id=inp,
                        dst_id=cn.node_id,
                        tensor_shape=shape_map.get(inp, TensorShape(dims=(1,))),
                    ))

        return graph

    def from_transformer_spec(
        self,
        *,
        name: str = "transformer",
        n_layers: int = 32,
        d_model: int = 4096,
        n_heads: int = 32,
        d_ff: int | None = None,
        vocab_size: int = 32000,
        seq_len: int = 2048,
    ) -> ComputeGraph:
        """Build a :class:`ComputeGraph` from a Transformer specification.

        This creates the canonical Transformer block structure:
        Embedding → N × (Attention + FFN) with residual connections.

        Parameters:
            name: Model name.
            n_layers: Number of Transformer layers.
            d_model: Hidden dimension.
            n_heads: Number of attention heads.
            d_ff: Feed-forward intermediate dimension (default 4 × d_model).
            vocab_size: Vocabulary size for the embedding layer.
            seq_len: Sequence length for shape estimation.
        """
        if d_ff is None:
            d_ff = 4 * d_model
        d_head = d_model // n_heads

        graph = ComputeGraph(name=name)

        # Parameter count estimate
        embed_params = vocab_size * d_model
        attn_params_per_layer = 4 * d_model * d_model  # Q, K, V, O projections
        ff_params_per_layer = 2 * d_model * d_ff  # up + down
        norm_params_per_layer = 2 * d_model  # LayerNorm
        total = embed_params + n_layers * (attn_params_per_layer + ff_params_per_layer + 2 * norm_params_per_layer)
        graph.total_parameters = total

        node_idx = 0
        prev_id = "input"

        # Embedding
        emb = ComputeNode(
            node_id=f"embed_{node_idx}",
            op_type=OpType.EMBEDDING,
            input_shapes=[TensorShape(dims=(seq_len,))],
            output_shape=TensorShape(dims=(seq_len, d_model)),
        )
        emb.flops = seq_len * d_model
        graph.nodes.append(emb)
        graph.edges.append(ComputeEdge(src_id=prev_id, dst_id=emb.node_id))
        prev_id = emb.node_id
        node_idx += 1

        for layer in range(n_layers):
            prefix = f"layer{layer}"

            # --- Attention block ---
            # LayerNorm
            ln1 = ComputeNode(
                node_id=f"{prefix}_ln1_{node_idx}",
                op_type=OpType.LAYERNORM,
                input_shapes=[TensorShape(dims=(seq_len, d_model))],
                output_shape=TensorShape(dims=(seq_len, d_model)),
            )
            ln1.estimate_flops()
            graph.nodes.append(ln1)
            graph.edges.append(ComputeEdge(src_id=prev_id, dst_id=ln1.node_id))
            node_idx += 1

            # QKV projection (single MatMul for simplicity)
            qkv = ComputeNode(
                node_id=f"{prefix}_qkv_{node_idx}",
                op_type=OpType.MATMUL,
                input_shapes=[
                    TensorShape(dims=(seq_len, d_model)),
                    TensorShape(dims=(d_model, 3 * d_model)),
                ],
                output_shape=TensorShape(dims=(seq_len, 3 * d_model)),
            )
            qkv.estimate_flops()
            graph.nodes.append(qkv)
            graph.edges.append(ComputeEdge(src_id=ln1.node_id, dst_id=qkv.node_id))
            node_idx += 1

            # Attention scores: Q @ K^T  → (n_heads, seq, seq)
            attn_score = ComputeNode(
                node_id=f"{prefix}_attn_score_{node_idx}",
                op_type=OpType.MATMUL,
                input_shapes=[
                    TensorShape(dims=(n_heads, seq_len, d_head)),
                    TensorShape(dims=(n_heads, d_head, seq_len)),
                ],
                output_shape=TensorShape(dims=(n_heads, seq_len, seq_len)),
            )
            attn_score.estimate_flops()
            graph.nodes.append(attn_score)
            graph.edges.append(ComputeEdge(src_id=qkv.node_id, dst_id=attn_score.node_id))
            node_idx += 1

            # Softmax
            sm = ComputeNode(
                node_id=f"{prefix}_softmax_{node_idx}",
                op_type=OpType.SOFTMAX,
                input_shapes=[TensorShape(dims=(n_heads, seq_len, seq_len))],
                output_shape=TensorShape(dims=(n_heads, seq_len, seq_len)),
            )
            sm.estimate_flops()
            graph.nodes.append(sm)
            graph.edges.append(ComputeEdge(src_id=attn_score.node_id, dst_id=sm.node_id))
            node_idx += 1

            # Attention × V  → (n_heads, seq, d_head)
            attn_v = ComputeNode(
                node_id=f"{prefix}_attn_v_{node_idx}",
                op_type=OpType.MATMUL,
                input_shapes=[
                    TensorShape(dims=(n_heads, seq_len, seq_len)),
                    TensorShape(dims=(n_heads, seq_len, d_head)),
                ],
                output_shape=TensorShape(dims=(n_heads, seq_len, d_head)),
            )
            attn_v.estimate_flops()
            graph.nodes.append(attn_v)
            graph.edges.append(ComputeEdge(src_id=sm.node_id, dst_id=attn_v.node_id))
            node_idx += 1

            # Output projection
            o_proj = ComputeNode(
                node_id=f"{prefix}_o_proj_{node_idx}",
                op_type=OpType.MATMUL,
                input_shapes=[
                    TensorShape(dims=(seq_len, d_model)),
                    TensorShape(dims=(d_model, d_model)),
                ],
                output_shape=TensorShape(dims=(seq_len, d_model)),
            )
            o_proj.estimate_flops()
            graph.nodes.append(o_proj)
            graph.edges.append(ComputeEdge(src_id=attn_v.node_id, dst_id=o_proj.node_id))
            node_idx += 1

            # Residual add
            res1 = ComputeNode(
                node_id=f"{prefix}_res1_{node_idx}",
                op_type=OpType.ADD,
                input_shapes=[
                    TensorShape(dims=(seq_len, d_model)),
                    TensorShape(dims=(seq_len, d_model)),
                ],
                output_shape=TensorShape(dims=(seq_len, d_model)),
            )
            res1.estimate_flops()
            graph.nodes.append(res1)
            graph.edges.append(ComputeEdge(src_id=o_proj.node_id, dst_id=res1.node_id))
            graph.edges.append(ComputeEdge(src_id=prev_id, dst_id=res1.node_id))
            node_idx += 1

            # --- FFN block ---
            ln2 = ComputeNode(
                node_id=f"{prefix}_ln2_{node_idx}",
                op_type=OpType.LAYERNORM,
                input_shapes=[TensorShape(dims=(seq_len, d_model))],
                output_shape=TensorShape(dims=(seq_len, d_model)),
            )
            ln2.estimate_flops()
            graph.nodes.append(ln2)
            graph.edges.append(ComputeEdge(src_id=res1.node_id, dst_id=ln2.node_id))
            node_idx += 1

            # FFN up
            ff_up = ComputeNode(
                node_id=f"{prefix}_ff_up_{node_idx}",
                op_type=OpType.MATMUL,
                input_shapes=[
                    TensorShape(dims=(seq_len, d_model)),
                    TensorShape(dims=(d_model, d_ff)),
                ],
                output_shape=TensorShape(dims=(seq_len, d_ff)),
            )
            ff_up.estimate_flops()
            graph.nodes.append(ff_up)
            graph.edges.append(ComputeEdge(src_id=ln2.node_id, dst_id=ff_up.node_id))
            node_idx += 1

            # Activation
            act = ComputeNode(
                node_id=f"{prefix}_gelu_{node_idx}",
                op_type=OpType.GELU,
                input_shapes=[TensorShape(dims=(seq_len, d_ff))],
                output_shape=TensorShape(dims=(seq_len, d_ff)),
            )
            act.estimate_flops()
            graph.nodes.append(act)
            graph.edges.append(ComputeEdge(src_id=ff_up.node_id, dst_id=act.node_id))
            node_idx += 1

            # FFN down
            ff_down = ComputeNode(
                node_id=f"{prefix}_ff_down_{node_idx}",
                op_type=OpType.MATMUL,
                input_shapes=[
                    TensorShape(dims=(seq_len, d_ff)),
                    TensorShape(dims=(d_ff, d_model)),
                ],
                output_shape=TensorShape(dims=(seq_len, d_model)),
            )
            ff_down.estimate_flops()
            graph.nodes.append(ff_down)
            graph.edges.append(ComputeEdge(src_id=act.node_id, dst_id=ff_down.node_id))
            node_idx += 1

            # Residual add
            res2 = ComputeNode(
                node_id=f"{prefix}_res2_{node_idx}",
                op_type=OpType.ADD,
                input_shapes=[
                    TensorShape(dims=(seq_len, d_model)),
                    TensorShape(dims=(seq_len, d_model)),
                ],
                output_shape=TensorShape(dims=(seq_len, d_model)),
            )
            res2.estimate_flops()
            graph.nodes.append(res2)
            graph.edges.append(ComputeEdge(src_id=ff_down.node_id, dst_id=res2.node_id))
            graph.edges.append(ComputeEdge(src_id=res1.node_id, dst_id=res2.node_id))
            node_idx += 1

            prev_id = res2.node_id

        return graph
