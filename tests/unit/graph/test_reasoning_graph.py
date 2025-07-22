from topology import ReasoningGraph


def test_from_yaml():
    g = ReasoningGraph.from_yaml("./tests/unit/graph/graph.yaml")
    assert g.edges == [
        [(3, 0), (3, 1), (3, 2)],
        [(0, 1), (1, 3), (2, 0), (2, 1)],
        [(1, 2)],
    ]
