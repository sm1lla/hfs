import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ..feature_selection import TSELSelector


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
    support = np.array([True, False, False, False, False])
    return (X, y, hierarchy, result, support)


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
    support = np.array([False, True, False, False, True])
    return (X, y, hierarchy, result, support)


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
    support = np.array([True, True, True, True, True])
    return (X, y, hierarchy, result, support)


@pytest.mark.parametrize(
    "data",
    [
        data1(),
        data2(),
        data2(),
    ],
)
def test_TSEL_selection(data):
    X, y, hierarchy, result, support = data
    selector = TSELSelector(hierarchy)
    selector.fit(X, y)
    X = selector.transform(X)
    assert np.array_equal(X, result)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)
