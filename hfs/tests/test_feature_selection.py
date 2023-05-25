import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ..feature_selection import TreeBasedFeatureSelector


@pytest.fixture
def data():
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
    edges = [("A", "B"), ("B", "C"), ("A", "D"), ("A", "E")]
    hierarchy = nx.DiGraph(edges)
    y = np.array([0, 0, 0, 0, 1])
    X = df.to_numpy()
    return (X, y, hierarchy, columns)


def test_tree_based_selection(data):
    X, y, hierarchy, columns = data
    selector = TreeBasedFeatureSelector(hierarchy, columns)
    selector.fit(X, y)
    X = selector.transform(X)
    assert np.array_equal(X, np.array([[0], [0], [0], [0], [1]]))
