import torch
import torch.nn as nn

from configs import ClosureMLPCfg


class ClosureMLP(nn.Module):
    def __init__(self, cfg: ClosureMLPCfg):
        super().__init__()
        self.graph_height = cfg.height
        self.graph_width = cfg.width
        self.register_buffer(
            "log1p_N", torch.log1p(torch.tensor(cfg.width, dtype=torch.float32))
        )
        self.register_buffer(
            "log1p_KN2",
            torch.log1p(torch.tensor(cfg.height * cfg.width**2, dtype=torch.float32)),
        )

        self.mlp = nn.Sequential(
            nn.Linear(3, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    # closure_vector: Size[num_nodes, 3]
    def forward(self, closure_vector):
        closure = closure_vector[:, 0]
        fanout = torch.log1p(closure_vector[:, 1]) / self.log1p_N
        subtree_edge_cnt = torch.log1p(closure_vector[:, 2]) / self.log1p_KN2

        closure_vector_scaled = torch.stack([closure, fanout, subtree_edge_cnt], dim=1)
        return self.mlp(closure_vector_scaled)
