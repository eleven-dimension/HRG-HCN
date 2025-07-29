import torch
import pytest

from topology import ReasoningGraph


def test_from_yaml():
    g = ReasoningGraph.from_yaml("./tests/unit/graph/graph.yaml")
    assert g.edges == [
        [(3, 0), (3, 1), (3, 2)],
        [(0, 1), (1, 3), (2, 0), (2, 1)],
        [(1, 2)],
    ]


@pytest.fixture
def example_graph():
    return ReasoningGraph(
        depth=4,
        width=4,
        root_index=1,
        edges=[
            [(1, 0), (1, 2), (1, 3)],  # layer 0 -> 1
            [(0, 1), (2, 0), (2, 3), (3, 1)],  # layer 1 -> 2
            [(1, 2)],  # layer 2 -> 3
        ],
    )


def test_subtree_size(example_graph):
    assert (
        example_graph.subtree_size_bit_map[0][1] == 8
    ), f"Expected subtree size: 8, got {example_graph.subtree_size_bit_map[0][1]}"


def test_to_tensor(example_graph):
    expected_depths = torch.tensor(
        [[0], [1], [1], [1], [2], [2], [2], [3]], dtype=torch.int32
    )
    expected_closure_vectors = torch.tensor(
        [
            [0, 3, 8],
            [0, 1, 2],
            [0, 2, 2],
            [0, 1, 2],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=torch.int32,
    )
    expected_edge_index = torch.tensor(
        [[0, 0, 0, 1, 2, 2, 3, 5], [1, 2, 3, 5, 4, 6, 5, 7]]
    )
    output_depths, output_closure_vectors, output_edge_index = example_graph.to_tensor()
    assert torch.equal(output_depths, expected_depths)
    assert torch.equal(output_closure_vectors, expected_closure_vectors)
    assert torch.equal(output_edge_index, expected_edge_index)
