from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Self
from collections import defaultdict
import yaml

# [parent_index, child_index]
Node = Tuple[int, int]
Edge = Tuple[Node, Node]


def construct_2d_list(value, width, depth):
    return [[value for _ in range(width)] for _ in range(depth)]


@dataclass
class ReasoningGraph:
    depth: int
    width: int
    root_index: int
    edges: List[List[Tuple[int, int]]]  # edges[i] = [(parent_index, child_index), ...]

    adjacency_list: Dict[Node, List[Node]] = field(init=False)

    nodes_bit_map: List[List[int]] = field(init=False)
    fanout_bit_map: List[List[int]] = field(init=False)
    subtree_size_bit_map: List[List[int]] = field(init=False)
    nodes: List[Node] = field(init=False)
    node_to_index: Dict[Node, int] = field(init=False)

    def __post_init__(self) -> None:
        self.adjacency_list = defaultdict(list)
        self.nodes_bit_map = construct_2d_list(
            value=0, width=self.width, depth=self.depth
        )
        self.fanout_bit_map = construct_2d_list(
            value=0, width=self.width, depth=self.depth
        )
        self.subtree_size_bit_map = construct_2d_list(
            value=0, width=self.width, depth=self.depth
        )

        self.nodes = []
        self.node_to_index = {}
        self.get_all_nodes_and_fanouts()
        self.get_subtree_size()

    def get_all_nodes_and_fanouts(self):
        for depth, current_edges in enumerate(self.edges):
            for parent_index, child_index in current_edges:
                # mark endpoints as present
                self.nodes_bit_map[depth][parent_index] = 1
                self.nodes_bit_map[depth + 1][child_index] = 1
                # accumulate fan‑out for parent
                self.fanout_bit_map[depth][parent_index] += 1
                # directed edge parent -> child
                self.adjacency_list[(depth, parent_index)].append(
                    (depth + 1, child_index)
                )

        # build flat list of nodes and an index lookup table
        for d in range(self.depth):
            for w in range(self.width):
                if self.nodes_bit_map[d][w]:
                    self.nodes.append((d, w))
        self.node_to_index = {node: i for i, node in enumerate(self.nodes)}

    def get_subtree_size(self) -> None:
        num_nodes = len(self.nodes)
        # fan‑out per global index
        fanouts: List[int] = [0] * num_nodes
        for index, (d, w) in enumerate(self.nodes):
            fanouts[index] = self.fanout_bit_map[d][w]

        reachable_nodes: List[int] = [0] * num_nodes  # bit‑set cache R[v]

        # process depths from the deepest layer up to the root layer
        for d in range(self.depth - 1, -1, -1):
            for w in range(self.width):
                if not self.nodes_bit_map[d][w]:
                    continue

                current_node = (d, w)
                current_node_index = self.node_to_index[current_node]

                bit_set = 0  # union of children's reachability plus children themselves
                for child in self.adjacency_list.get(current_node, []):
                    child_index = self.node_to_index[child]
                    bit_set |= reachable_nodes[child_index]  # descendants
                    bit_set |= 1 << child_index  # the child itself

                reachable_nodes[current_node_index] = bit_set

                # compute subtree edge count
                subtree_edges_cnt = fanouts[current_node_index]
                temp = bit_set
                while temp:
                    lowbit = temp & -temp
                    descendant_index = lowbit.bit_length() - 1
                    subtree_edges_cnt += fanouts[descendant_index]
                    temp ^= lowbit  # clear bit and continue

                self.subtree_size_bit_map[d][w] = subtree_edges_cnt

    @classmethod
    def from_yaml(cls, filepath: str) -> Self:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        edges = []
        if data.get("edges") is not None:
            edges = [[tuple(edge) for edge in edge_list] for edge_list in data["edges"]]
        return cls(data["max_depth"], data["agent_num"], data["root_index"], edges)


if __name__ == "__main__":
    g = ReasoningGraph(
        depth=4,
        width=4,
        root_index=1,
        edges=[
            [(1, 0), (1, 2), (1, 3)],  # layer 0 -> 1
            [(0, 1), (2, 0), (2, 3), (3, 1)],  # layer 1 -> 2
            [(1, 2)],  # layer 2 -> 3
        ],
    )
    print("Adjacency list:", g.adjacency_list)
    print("Subtree edge counts:")
    for row in g.subtree_size_bit_map:
        print(row)
