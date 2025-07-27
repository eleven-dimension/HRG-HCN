import torch

from layers import InvEdgeEncoder
from configs import InvEdgeEncoderCfg


def test_edge_encoder():
    cfg = InvEdgeEncoderCfg(width=5, embedding_dim=32, hidden_dim=32, output_dim=16)
    encoder = InvEdgeEncoder(cfg)
    encoder.eval()
    parent_ids = torch.randint(0, 5, (17,))
    children_ids = torch.randint(0, 5, (17,))
    with torch.no_grad():
        embedding = encoder(parent_ids, children_ids).detach().cpu().numpy()
    assert embedding.shape == torch.Size(
        (17, 16)
    ), f"Expected shape Size(17, 16), got {embedding.shape}"
