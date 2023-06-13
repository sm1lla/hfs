import networkx as nx
import numpy as np
import pytest

from ..preprocessing import HierarchicalPreprocessor


@pytest.fixture
def data():
    X = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0]])

    edges = [("A", "B"), ("A", "C"), ("C", "D"), ("E", "A")]
    hierarchy = nx.to_numpy_array(nx.DiGraph(edges))

    X_transformed = np.array([[1, 0, 1, 0, 1], [1, 1, 0, 0, 1], [1, 1, 0, 0, 1]])

    return (X, X_transformed, hierarchy)


def test_HP(data):
    X, X_transformed, hierarchy = data

    preprocessor = HierarchicalPreprocessor(hierarchy)

    preprocessor.fit(X)
    X = preprocessor.transform(X)
    assert np.array_equal(X, X_transformed)
