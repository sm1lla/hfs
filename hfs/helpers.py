import networkx as nx
import numpy as np
from networkx.algorithms.simple_paths import all_simple_paths


def create_feature_tree(hierarchy: nx.DiGraph) -> nx.DiGraph:
    # create parent node to join hierarchies
    roots = [x for x in hierarchy.nodes() if hierarchy.in_degree(x) == 0]

    for root_node in roots:
        hierarchy.add_edge("ROOT", root_node)

    return hierarchy


def get_paths(graph: nx.DiGraph):
    leafs = [
        node
        for node in graph
        if graph.in_degree(node) > 0 and graph.out_degree(node) == 0
    ]

    paths = all_simple_paths(graph, "ROOT", leafs)
    return paths


def lift(data: np.ndarray, labels: np.ndarray):
    """returns list including lift value for each feature"""
    lift_values = []
    num_samples, num_features = data.shape

    for index in range(num_features):
        column = data[:, index]
        prob_feature = np.count_nonzero(column) / num_samples
        prob_event_conditional = len(
            [
                value
                for index, value in enumerate(column)
                if value != 0 and labels[index] != 0
            ]
        ) / len(np.nonzero(column))

        lift_values.append(prob_event_conditional / prob_feature)
    return lift_values
