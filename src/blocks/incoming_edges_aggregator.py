import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from layers import InvEdgeEncoder
from configs import IncomingEdgesAggregatorCfg


class IncomingEdgesAggregator(nn.Module):
    def __init__(self, cfg: IncomingEdgesAggregatorCfg):
        super().__init__()
        self.edge_encoder = InvEdgeEncoder(cfg.edge_encoder)

    def forward(self, num_nodes: int, edge_index):
        edge_embeddings = self.edge_encoder(edge_index[0], edge_index[1])
        target_node_indices = edge_index[1]
        return scatter_mean(
            edge_embeddings, target_node_indices, dim=0, dim_size=num_nodes
        )
