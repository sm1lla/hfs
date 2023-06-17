import os
import networkx as nx
import numpy as np
import pytest


from ..preprocessing import HierarchicalPreprocessor

@pytest.fixture
def data1():
    X = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0]])

    edges = [("A", "B"), ("A", "C"), ("C", "D"), ("E", "A")]
    hierarchy = nx.to_numpy_array(nx.DiGraph(edges))

    X_transformed = np.array([[1, 0, 1, 0, 1], [1, 1, 0, 0, 1], [1, 1, 0, 0, 1]])

    return (X, X_transformed, hierarchy)

@pytest.fixture
def data2():
    
    edges = [("GO:2001090", "GO:2001091"),("GO:2001090", "GO:2001092"),("GO:2001091", "GO:2001093")
             ,("GO:2001091", "GO:2001094"),("GO:2001093", "GO:2001095")]
    #      0
    #   1      2
    #3    4
    #5
    hierarchy = nx.to_numpy_array(nx.DiGraph(edges))
    X_identifiers = list([0,1,2,4])
    X = np.ones((2,len(X_identifiers)))
    # in X there is 0,1,2,4
    edges_transformed = [("GO:2001090", "GO:2001091"),("GO:2001090", "GO:2001092"),
                ("GO:2001091", "GO:2001094")]
    h = nx.DiGraph(edges_transformed)

    h.add_edge("ROOT", "GO:2001090")
    
    hierarchy_transformed = nx.to_numpy_array(h)

    return (X,  hierarchy, hierarchy_transformed, X_identifiers)


def test_HP(data1):
    X, X_transformed, hierarchy = data1

    preprocessor = HierarchicalPreprocessor(hierarchy)

    preprocessor.fit(X)
    X = preprocessor.transform(X)
    assert np.array_equal(X, X_transformed)

def test_shrink_dag(data2):
    X, hierarchy, hierarchy_transformed, X_identifiers = data2
    preprocessor = HierarchicalPreprocessor(hierarchy)
    preprocessor.fit(X, columns=X_identifiers)
    hierarchy = preprocessor.get_hierarchy()
    assert np.equal(hierarchy.all(), hierarchy_transformed.all())
    
