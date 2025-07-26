from dataclasses import dataclass, field
from omegaconf import MISSING


# 0.  Mix‑ins for graph dimensions (shared)
@dataclass
class EmbeddingCfg:
    dim: int = MISSING


@dataclass
class GraphWidthMixin:
    graph_width: int = MISSING


@dataclass
class GraphHeightMixin:
    graph_height: int = MISSING


@dataclass
class GraphGeometryMixin(GraphWidthMixin, GraphHeightMixin):
    pass


# 1.  Leaf‑level layer configs
@dataclass
class MonotonicFunctionCfg(GraphHeightMixin):
    epsilon: float = 1e-6


@dataclass
class GatedNodeAggregatorCfg:
    feature_dim: int = MISSING
    hidden_dim: int = MISSING


@dataclass
class ClosureMLPCfg(GraphGeometryMixin):
    hidden_dim: int = MISSING
    output_dim: int = MISSING


@dataclass
class InvEdgeEncoderCfg(GraphWidthMixin):
    embedding_dim: int = MISSING
    hidden_dim: int = MISSING
    output_dim: int = MISSING


# 2.  Block‑level (a stack of layers)
@dataclass
class IncomingEdgesAggregatorCfg(GraphWidthMixin):
    edge_encoder: InvEdgeEncoderCfg = field(default_factory=InvEdgeEncoderCfg)


@dataclass
class NodeEncoderCfg:
    embedding: EmbeddingCfg = field(default_factory=EmbeddingCfg)
    monotonic_function: MonotonicFunctionCfg = field(
        default_factory=MonotonicFunctionCfg
    )
    node_aggregator: GatedNodeAggregatorCfg = field(
        default_factory=GatedNodeAggregatorCfg
    )
    closure_mlp: ClosureMLPCfg = field(default_factory=ClosureMLPCfg)
    incoming_edges_aggregator: IncomingEdgesAggregatorCfg = field(
        default_factory=IncomingEdgesAggregatorCfg
    )


# 3.  Model‑level  (top of the hierarchy)


@dataclass
class GraphCfg(GraphGeometryMixin):
    pass


# @dataclass
# class GCNConfig:
#     """What `NodeEncoder` will receive."""

#     graph: GraphCfg = GraphCfg()
#     embedding: EmbeddingCfg = EmbeddingCfg()
#     layer: LayerCfg = LayerCfg()
