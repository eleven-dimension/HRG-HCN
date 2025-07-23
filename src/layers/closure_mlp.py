import torch
import torch.nn as nn


class ClosureMLP(nn.Module):
    def __init__(
        self, num_agents: int, num_layers: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()
        self.num_agents = num_agents
        self.num_layers = num_layers
        self.register_buffer(
            "log1p_N", torch.log1p(torch.tensor(num_agents, dtype=torch.float32))
        )
        self.register_buffer(
            "log1p_KN2",
            torch.log1p(torch.tensor(num_layers * num_agents**2, dtype=torch.float32)),
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
