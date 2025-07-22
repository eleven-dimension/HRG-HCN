from typing import List, Tuple, Self
import yaml

# [parent_index, child_index]
Edge = Tuple[int, int]


# Embedding:
# x: [num_nodes, feat_dim]
# edge_index_up: [2, num_edges]  child => parent edges
# edge_index_down: [2, num_edges]  parent => child edges
# depth: [num_nodes]
class ReasoningGraph:
    def __init__(
        self,
        max_depth: int,
        agent_num: int,
        root_index: int,
        edges: List[List[Edge]] | None,
    ):
        self.max_depth = max_depth
        self.agent_num = agent_num
        self.root_index = root_index
        self.edges = edges

    @classmethod
    def from_yaml(cls, filepath: str) -> Self:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        edges = None
        if data["edges"] is not None:
            edges = [[tuple(edge) for edge in edge_list] for edge_list in data["edges"]]
        return cls(data["max_depth"], data["agent_num"], data["root_index"], edges)
