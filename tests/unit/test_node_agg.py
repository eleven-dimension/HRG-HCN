import torch

from layers import GatedNodeAggregator
from configs import GatedNodeAggregatorCfg


def test_node_agg():
    cfg = GatedNodeAggregatorCfg(feature_dim=16, hidden_dim=32)
    agg = GatedNodeAggregator(cfg)
    agg.eval()
    x = torch.rand((7, 16))
    with torch.no_grad():
        embedding = agg(x, x, x).detach().cpu().numpy()
    assert embedding.shape == torch.Size(
        (7, 16)
    ), f"Expected shape Size(7, 16), got {embedding.shape}"
