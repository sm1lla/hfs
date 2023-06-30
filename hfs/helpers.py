import math
from fractions import Fraction

import networkx as nx
import numpy as np
from info_gain.info_gain import info_gain, info_gain_ratio
from networkx.algorithms.simple_paths import all_simple_paths
from numpy.linalg import norm
from pyitlib import discrete_random_variable as drv
from scipy import sparse


def getRelevance(xdata, ydata, node):
    """
    Gather relevance for a given node.

    Parameters
    ----------
    node
        Node for which the relevance should be obtained.
    xdata
        xdata
    ydata
        data as np array
    """
    p1 = (
        Fraction(
            xdata[(xdata[:, node] == 1) & (ydata == 1)].shape[0],
            xdata[(xdata[:, node] == 1)].shape[0],
        )
        if xdata[(xdata[:, node] == 1)].shape[0] != 0
        else 0
    )
    p2 = (
        Fraction(
            xdata[(xdata[:, node] == 0) & (ydata == 1)].shape[0],
            xdata[(xdata[:, node] == 0)].shape[0],
        )
        if xdata[(xdata[:, node] == 0)].shape[0] != 0
        else 0
    )
    p3 = 1 - p1
    p4 = 1 - p2

    rel = (p1 - p2) ** 2 + (p3 - p4) ** 2
    return rel


def checkData(dag, x_data, y_data):  # possible: bool checking
    data = np.column_stack((x_data, y_data))
    edges = list(nx.edge_dfs(dag, source=0, orientation="original"))
    for edge in edges:
        for idx in range(len(data)):
            if data[idx, edge[0]] == 0 and data[idx, edge[1]] == 1:
                # depending on number of errors -> delete those instances instead of throwing errors?
                raise ValueError(
                    f"Test instance {idx} violates 0-1-propagation on edge ({edge[0]}, {edge[1]})"
                    f"{data[idx]}"
                )


def expand_data(x_data, x_identifier, hierarchy, graph_identifier):
    pass


def get_irrelevant_leaves(x_identifier, digraph):
    return [
        x
        for x in digraph.nodes()
        if digraph.out_degree(x) == 0
        and digraph.in_degree(x) == 1
        and x not in x_identifier
        and x != "ROOT"
    ]


def get_leaves(graph: nx.DiGraph):
    return [
        node
        for node in graph
        if graph.in_degree(node) > 0 and graph.out_degree(node) == 0
    ]


def shrink_dag(x_identifier, digraph):
    leaves = get_irrelevant_leaves(x_identifier=x_identifier, digraph=digraph)
    while leaves:
        for x in leaves:
            digraph.remove_node(x)
        leaves = get_irrelevant_leaves(x_identifier=x_identifier, digraph=digraph)
    return digraph


def connect_dag(x_identifiers, digraph):
    top_sort = nx.topological_sort(digraph)

    # connect every node with at least one ancestor on each path that is for shure in x_i
    # i = 0: source is either in or not in, as they are no predecessors,
    # there should not be any artificial edge
    # i: for each pred there is a direct edge to the pred and iff pred not in x_ide
    #       also to their pred2. (it does not matter if pred2 is really in x, if it is not,
    #       the edge will be removed later anyway)
    # i+1: if i is in -> no artificial edge on this path needed
    #       if i is not -> artifical edge to every pred of i, so each path going through i
    #       will be continued, if i is removed later

    new_graph = digraph.copy()

    for node in list(top_sort):
        preds = list(digraph.predecessors(node))
        for pred in preds:
            new_connections = []
            if pred in x_identifiers:
                # to do -> make more efficient
                # sort x_identifiers according to order in digraph
                # x_identifiers.sort(key = lambda i: top_sort.index())
                pass
            else:
                for pred_of_pred in digraph.predecessors(pred):
                    new_connections.append(pred_of_pred)
                for new_connection in new_connections:
                    digraph.add_edge(new_connection, node)

    # remove all nodes (and edges) that are not in x_identifier
    x_identifiers_set = set(x_identifiers)
    nodes_to_remove = [node for node in digraph.nodes if node not in x_identifiers_set]
    digraph.remove_nodes_from(nodes_to_remove)

    return digraph


def create_feature_tree(hierarchy: nx.DiGraph) -> nx.DiGraph:
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


# TODO: add file for metrics
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
        ig = info_gain(data[:, column_index], labels)
        ig_values.append(ig)
    return ig_values


def get_columns_for_numpy_hierarchy(hierarchy: nx.DiGraph, num_columns: int):
    """If each node in the hierarchy is named after a column's index this methods will give you
    the mapping from column index to node name of the node after the graph was transformed to a numpy array
    and back
    Value represents new node (numpy) while index is the name of node in Digraph before transformation.
    """
    columns = []
    for node in range(num_columns):
        columns.append(
            list(hierarchy.nodes()).index(node) if node in hierarchy.nodes else -1
        )
    return columns


def normalize_score(score, max_value):
    if score != 0:
        score = math.log(1 + (score / max_value)) + 1
    return score


def conditional_mutual_information(node1, node2, y):
    return drv.information_mutual_conditional(node1, node2, y)


def cosine_similarity(i: np.ndarray, j: np.ndarray):
    return np.dot(i, j) / (norm(i) * norm(j))


def gain_ratio(data, labels):
    gr_values = []
    for column_index in range(data.shape[1]):
        gr = info_gain_ratio(data[:, column_index], labels)
        gr_values.append(gr)
    return gr_values


def pearson_correlation(x: np.array, y: np.array):
    return np.corrcoef(x, y)[0, 1]


def compute_aggregated_values(node: int, X, hierarchy: nx.DiGraph, columns: list[int]):
    if hierarchy.out_degree(node) == 0:
        return X
    else:
        children = hierarchy.successors(node)
        aggregated = np.zeros((X.shape[0], 1))
        for child in children:
            X = compute_aggregated_values(child)
            aggregated = np.add(aggregated, X[:, columns.index(child)])
    if node != "ROOT":
        column_index = columns.index(node)
        X[:, column_index] = aggregated
    return X
