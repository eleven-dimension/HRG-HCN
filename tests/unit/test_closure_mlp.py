import torch

from layers import ClosureMLP
from configs import ClosureMLPCfg


def test_closure_mlp():
    cfg = ClosureMLPCfg(width=5, height=7, hidden_dim=32, output_dim=16)
    closure_mlp = ClosureMLP(cfg)
    closure_mlp.eval()
    x = torch.randint(0, 11, (7, 3))
    with torch.no_grad():
        embedding = closure_mlp(x).detach().cpu().numpy()
    assert embedding.shape == torch.Size(
        (7, 16)
    ), f"Expected shape Size(7, 16), got {embedding.shape}"
