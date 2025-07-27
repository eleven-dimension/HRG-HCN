import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GatedNodeAggregator, MonotonicFunction, ClosureMLP
from manifolds import logmap0, expmap
from configs import NodeEncoderCfg

from .incoming_edges_aggregator import IncomingEdgesAggregator


class NodeEncoder(nn.Module):
    def __init__(self, cfg: NodeEncoderCfg) -> None:
        super().__init__()
        self.embedding_dim = cfg.embedding_dim
        self.depth_encoder = MonotonicFunction(cfg.monotonic_function)
        self.closure_mlp = ClosureMLP(cfg.closure_mlp)
        self.incoming_edges_aggregator = IncomingEdgesAggregator(
            cfg.incoming_edges_aggregator
        )
        self.node_aggregator = GatedNodeAggregator(cfg.node_aggregator)

        depth_unit_tangent_vector = F.one_hot(
            torch.tensor(1), num_classes=cfg.embedding_dim
        ).float()
        self.register_buffer("depth_unit_tangent_vector", depth_unit_tangent_vector)

    def forward(self, depths, closure_vectors, edge_index):
        # depths: Size[num_nodes, 1]
        # closure_vectors: Size[num_nodes, 3]
        # edge_index: Size[2, num_edges]

        # Radial depth embedding
        monotonic_radiuses = self.depth_encoder()  # Size[graph_height + 1]
        depth_scalars = torch.gather(
            monotonic_radiuses, dim=0, index=depths.squeeze(-1)
        ).unsqueeze(
            -1
        )  # Size[num_nodes, 1]
        depth_tangent_vectors = (
            depth_scalars * self.depth_unit_tangent_vector.unsqueeze(0)
        )  # Size[num_nodes, embedding_dim]

        # Closure embedding
        closure_tangent_embeddings = self.closure_mlp(closure_vectors)

        # Incoming edge embedding
        num_nodes = depths.size(0)
        incoming_edges_embeddings = self.incoming_edges_aggregator(
            num_nodes, edge_index
        )  # Size[num_nodes, embedding_dim]

        # Aggregation
        return self.node_aggregator(
            depth_tangent_vectors, closure_tangent_embeddings, incoming_edges_embeddings
        )
