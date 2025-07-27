import torch

from blocks import NodeEncoder
from configs import (
    NodeEncoderCfg,
    MonotonicFunctionCfg,
    GatedNodeAggregatorCfg,
    ClosureMLPCfg,
    InvEdgeEncoderCfg,
    IncomingEdgesAggregatorCfg,
)


def test_node_encoder():
    width = 8
    height = 5
    cfg = NodeEncoderCfg(
        embedding_dim=16,
        monotonic_function=MonotonicFunctionCfg(height),
        node_aggregator=GatedNodeAggregatorCfg(feature_dim=16, hidden_dim=32),
        closure_mlp=ClosureMLPCfg(width, height, hidden_dim=32, output_dim=16),
        incoming_edges_aggregator=IncomingEdgesAggregatorCfg(
            width=4,
            edge_encoder=InvEdgeEncoderCfg(
                width, embedding_dim=32, hidden_dim=32, output_dim=16
            ),
        ),
    )

    encoder = NodeEncoder(cfg)
    encoder.eval()
    # inputs
    depths = torch.randint(0, 2, size=(17, 1))
    closure_vector = torch.randint(0, 40, size=(17, 3))
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 3, 4], [2, 3, 2, 3, 0, 1, 1]])
    with torch.no_grad():
        node_embeddings = encoder(depths, closure_vector, edge_index)

    assert node_embeddings.shape == torch.Size(
        (17, 16)
    ), f"Expected shape Size(17, 16), got {node_embeddings.shape}"
