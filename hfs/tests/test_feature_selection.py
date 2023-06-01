import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ..feature_selection import TreeBasedFeatureSelector


def data1():
    columns = ["A", "B", "C", "D", "E"]
    df = pd.DataFrame(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        columns=columns,
    )
    edges = [(0, 1), (1, 2), (0, 3), (0, 4)]
    hierarchy = nx.to_numpy_array(nx.DiGraph(edges))
    y = np.array([0, 0, 0, 0, 1])
    X = df.to_numpy()
    result = np.array([[0], [0], [0], [0], [1]])
    return (X, y, hierarchy, result)


def data2():
    X = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
    )
    edges = [(0, 1), (1, 2), (2, 3), (0, 4)]
    hierarchy = nx.to_numpy_array(nx.DiGraph(edges))
    y = np.array([1, 0, 0, 1, 1])
    result = np.array(
        [
            [1, 1],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 0],
        ]
    )

    return (X, y, hierarchy, result)


def data3():
    X = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
    )

    hierarchy = None
    y = np.array([1, 0, 0, 1, 1])
    result = X
    return (X, y, hierarchy, result)


@pytest.fixture
def data():
    return [data1(), data2(), data3()]


def test_tree_based_selection(data):
    for example in data:
        X, y, hierarchy, result = example
        selector = TreeBasedFeatureSelector(hierarchy)
        selector.fit(X, y)
        X = selector.transform(X)
        assert np.array_equal(X, result)
