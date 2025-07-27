import torch

from blocks import IncomingEdgesAggregator
from configs import IncomingEdgesAggregatorCfg, InvEdgeEncoderCfg


def test_incoming_edges_agg():
    cfg = IncomingEdgesAggregatorCfg(
        width=5,
        edge_encoder=InvEdgeEncoderCfg(
            width=5, embedding_dim=32, hidden_dim=32, output_dim=16
        ),
    )

    incoming_edges_agg = IncomingEdgesAggregator(cfg)
    incoming_edges_agg.eval()

    with torch.no_grad():
        incoming_edges_embedding = incoming_edges_agg(
            5, torch.tensor([[0, 0, 1, 1, 2, 3, 4], [2, 3, 2, 3, 0, 1, 1]])
        )
    assert incoming_edges_embedding.shape == torch.Size((5, 16))
