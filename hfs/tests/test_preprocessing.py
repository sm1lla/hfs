import os
import networkx as nx
import numpy as np
import pytest

import sys
sys.path.append('/home/kathrin/hfs/hfs/')
from base import HierarchicalEstimator

from go import open_dag

from preprocessing import HierarchicalPreprocessor

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
    X_identifiers = list(["GO:2001090", "GO:2001091", "GO:2001092", "GO:2001094"])
    X = np.ones([len(X_identifiers),2])
    edges_transformed = [("GO:2001090", "GO:2001091"),("GO:2001090", "GO:2001092"),
                ("GO:2001091", "GO:2001094")]
    hierarchy_transformed = nx.to_numpy_array(nx.DiGraph(edges_transformed))

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
    preprocessor.fit(X, column_names=X_identifiers)
    hierarchy = preprocessor.get_hierarchy()
    assert np.equal(hierarchy, hierarchy_transformed)
    
