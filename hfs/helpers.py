import networkx as nx
import numpy as np
from info_gain.info_gain import info_gain
from networkx.algorithms.simple_paths import all_simple_paths
from scipy import sparse


def create_feature_tree(hierarchy: nx.DiGraph, column_names: list[str]) -> nx.DiGraph:
    # add missing nodes to hierarchy
    for column in column_names:
        if column not in hierarchy.nodes():
            hierarchy.add_node(column)
    roots = [x for x in hierarchy.nodes() if hierarchy.in_degree(x) == 0]
    # create parent node to join hierarchies
    for root_node in roots:
        hierarchy.add_edge("ROOT", root_node)
    if not roots:
        hierarchy.add_node("ROOT")

    return hierarchy


def get_paths(graph: nx.DiGraph, reverse=False):
    leaves = get_leaves(graph)
    paths = list(all_simple_paths(graph, "ROOT", leaves))
    if reverse:
        for path in paths:
            path.reverse()
    return paths


def lift(data, labels):
    """returns list including lift value for each feature"""
    lift_values = []
    num_samples, num_features = data.shape

    for index in range(num_features):
        # deal with sparse matrices
        if sparse.issparse(data):
            data = data.tocsr()
            column = data[:, index]
            non_zero_values = column.size
        else:
            column = data[:, index]
            non_zero_values = np.count_nonzero(column)

        prob_feature = non_zero_values / num_samples

        prob_event_conditional = (
            len(
                [
                    value
                    for index, value in enumerate(column)
                    if value != 0 and labels[index] != 0
                ]
            )
            / non_zero_values
        )

        lift_values.append(prob_event_conditional / prob_feature)
    return lift_values


def information_gain(data, labels):
    ig_values = []
    for column_index in range(data.shape[1]):
        ig = info_gain(labels, data[:, column_index])
        ig_values.append(ig)
    return ig_values


def get_leaves(graph: nx.DiGraph):
    return [
        node
        for node in graph
        if graph.in_degree(node) > 0 and graph.out_degree(node) == 0
    ]
