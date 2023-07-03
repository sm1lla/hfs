import os

import networkx as nx
import numpy as np
import pytest

from ..data.data_utils import create_mapping_columns_to_nodes, load_data
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


@pytest.fixture
def data3():
    edges = [
        ("GO:2001090", "GO:2001091"),
        ("GO:2001090", "GO:2001092"),
        ("GO:2001091", "GO:2001093"),
        ("GO:2001091", "GO:2001094"),
        ("GO:2001093", "GO:2001095"),
    ]
    #      0
    #   1      2
    # 3    4
    # 5
    hierarchy = nx.to_numpy_array(nx.DiGraph(edges))
    X_identifiers = list([0, 1, 2, 4])
    X = np.ones((2, len(X_identifiers)))
    # in X there is 0,1,2,4
    edges_transformed = [
        ("GO:2001090", "GO:2001091"),
        ("GO:2001090", "GO:2001092"),
        ("GO:2001091", "GO:2001094"),
    ]
    h = nx.DiGraph(edges_transformed)

    h.add_edge("ROOT", "GO:2001090")

    hierarchy_transformed = nx.to_numpy_array(h)

    return (X, hierarchy, hierarchy_transformed, X_identifiers)


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


def test_shrink_dag(data3):
    X, hierarchy, hierarchy_transformed, X_identifiers = data3
    preprocessor = HierarchicalPreprocessor(hierarchy)
    preprocessor.fit(X, columns=X_identifiers)
    hierarchy = preprocessor.get_hierarchy()
    assert np.equal(hierarchy.all(), hierarchy_transformed.all())


def test_preprocessor_real_data():
    X, _, hierarchy = load_data()
    columns = create_mapping_columns_to_nodes(X, hierarchy)
    X = X.to_numpy()
    hierarchy = nx.to_numpy_array(hierarchy)
    preprocessor = HierarchicalPreprocessor(hierarchy)
    preprocessor.fit(X, columns)
    X_transformed = preprocessor.transform(X)
    hierarchy_updated = preprocessor.get_hierarchy()
    columns_updated = preprocessor.get_columns()
    assert X_transformed.shape[1] == len(columns_updated)
    assert hierarchy_updated.shape[1] == X_transformed.shape[1]
