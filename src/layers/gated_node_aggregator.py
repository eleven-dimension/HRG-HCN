import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import GatedNodeAggregatorCfg


class GatedNodeAggregator(nn.Module):
    def __init__(self, cfg: GatedNodeAggregatorCfg):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * cfg.feature_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 3),
        )

    def forward(self, node_embedding, closure_embedding, incoming_edges_embedding):
        concat = torch.cat(
            [node_embedding, closure_embedding, incoming_edges_embedding], dim=-1
        )  # shape: [num_nodes, 3 * feature_dim]

        gates = self.mlp(concat)
        gates = F.softmax(gates, dim=-1)  # shape: [num_nodes, 3]

        output = (
            gates[:, 0:1] * node_embedding
            + gates[:, 1:2] * closure_embedding
            + gates[:, 2:3] * incoming_edges_embedding
        )
        return output
