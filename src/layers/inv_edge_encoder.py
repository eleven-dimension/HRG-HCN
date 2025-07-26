import torch
import torch.nn as nn

from configs import InvEdgeEncoderCfg


# layer-invariant
class InvEdgeEncoder(nn.Module):
    def __init__(self, cfg: InvEdgeEncoderCfg):
        super().__init__()
        self.parent_embedding_layer = nn.Embedding(cfg.graph_width, cfg.embedding_dim)
        self.child_embedding_layer = nn.Embedding(cfg.graph_width, cfg.embedding_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(cfg.embedding_dim * 2, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    def forward(self, parent_ids, child_ids):
        parent_embedding = self.parent_embedding_layer(parent_ids)
        child_embedding = self.child_embedding_layer(child_ids)
        edge_input = torch.cat([parent_embedding, child_embedding], dim=-1)

        return self.edge_mlp(edge_input)  # shape: (num_edges, output_dim)
