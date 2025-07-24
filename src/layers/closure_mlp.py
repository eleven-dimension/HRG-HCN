import torch
import torch.nn as nn


class ClosureMLP(nn.Module):
    def __init__(
        self, graph_height: int, graph_width: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        self.graph_height = graph_height
        self.graph_width = graph_width
        self.register_buffer(
            "log1p_N", torch.log1p(torch.tensor(graph_width, dtype=torch.float32))
        )
        self.register_buffer(
            "log1p_KN2",
            torch.log1p(
                torch.tensor(graph_height * graph_width**2, dtype=torch.float32)
            ),
        )

        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    # x: shape: (batch_size, 3)
    def forward(self, x):
        closure = x[:, 0]
        fanout = torch.log1p(x[:, 1]) / self.log1p_N
        subtree_edge_cnt = torch.log1p(x[:, 2]) / self.log1p_KN2

        x_scaled = torch.stack([closure, fanout, subtree_edge_cnt], dim=1)
        return self.mlp(x_scaled)
