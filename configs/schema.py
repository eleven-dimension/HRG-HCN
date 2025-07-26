from dataclasses import dataclass, field
from omegaconf import MISSING


# 1.  Leaf‑level layer configs
@dataclass
class InvEdgeEncoderCfg:
    width: int = MISSING
    embedding_dim: int = MISSING
    hidden_dim: int = MISSING
    output_dim: int = MISSING


@dataclass
class MonotonicFunctionCfg:
    height: int = MISSING
    epsilon: float = 1e-6


@dataclass
class GatedNodeAggregatorCfg:
    feature_dim: int = MISSING
    hidden_dim: int = MISSING


@dataclass
class ClosureMLPCfg:
    width: int = MISSING
    height: int = MISSING
    hidden_dim: int = MISSING
    output_dim: int = MISSING


# 2.  Block‑level configs
@dataclass
class IncomingEdgesAggregatorCfg:
    width: int = MISSING
    edge_encoder: InvEdgeEncoderCfg = field(default_factory=InvEdgeEncoderCfg)


@dataclass
class NodeEncoderCfg:
    embedding_dim: int = MISSING
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


# 3.  Model‑level config (root)
@dataclass
class ModelCfg:
    width: int = MISSING
    height: int = MISSING
    embedding_dim: int = MISSING

    node_encoder: NodeEncoderCfg = field(default_factory=NodeEncoderCfg)
