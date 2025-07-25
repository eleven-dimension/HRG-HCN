import torch
import torch.nn as nn


# layer-invariant
class InvEdgeEncoder(nn.Module):
    def __init__(
        self, graph_width: int, embedding_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        self.parent_embedding_layer = nn.Embedding(graph_width, embedding_dim)
        self.child_embedding_layer = nn.Embedding(graph_width, embedding_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, parent_ids, child_ids):
        parent_embedding = self.parent_embedding_layer(parent_ids)
        child_embedding = self.child_embedding_layer(child_ids)
        edge_input = torch.cat([parent_embedding, child_embedding], dim=-1)

        return self.edge_mlp(edge_input)  # shape: (num_edges, output_dim)
