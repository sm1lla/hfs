import networkx as nx
import numpy as np
import pytest

from ..helpers import get_columns_for_numpy_hierarchy
from ..preprocessing import HierarchicalPreprocessor


def data1():
    X = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    edges = [(1, 3), (3, 2), (0, 4), (0, 1)]
    hierarchy_original = nx.DiGraph(edges)
    columns = get_columns_for_numpy_hierarchy(hierarchy_original, X.shape[1])
    hierarchy_original = nx.to_numpy_array(hierarchy_original)
    hierarchy_transformed = nx.to_numpy_array(nx.DiGraph([(1, 3), (3, 2), (0, 1)]))
    X_transformed = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]])

    return (X, X_transformed, hierarchy_original, columns, hierarchy_transformed)


def data2():
    X = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0]])

    edges = [(1, 2), (1, 3), (3, 4), (0, 1)]
    hierarchy_original = nx.DiGraph(edges)
    columns = get_columns_for_numpy_hierarchy(hierarchy_original, X.shape[1])
    hierarchy_original = nx.to_numpy_array(hierarchy_original)
    edges_tranformed = [(1, 2), (0, 1)]
    hierarchy_transformed = nx.to_numpy_array(nx.DiGraph(edges_tranformed))
    X_transformed = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0]])

    return (X, X_transformed, hierarchy_original, columns, hierarchy_transformed)


@pytest.mark.parametrize(
    "data",
    [data1(), data2()],
)
def test_HP(data):
    X, X_transformed, hierarchy, columns, hierarchy_expected = data

    preprocessor = HierarchicalPreprocessor(hierarchy)

    preprocessor.fit(X, columns=columns)
    assert preprocessor.is_fitted_
    X = preprocessor.transform(X)
    assert np.array_equal(X, X_transformed)
    hierarchy_transformed = preprocessor.get_hierarchy()
    assert np.array_equal(hierarchy_transformed, hierarchy_expected)
