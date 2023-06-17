import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator


import sys
sys.path.append('/home/kathrin/hfs/hfs/')
from mrt import MRT
from hip import HIP
from filter import Filter
from ..hnb import HNB
from ..hnbs import HNBs
from ..rnb import RNB
from .fixtures.fixtures import *

import pytest

@pytest.fixture
def data1():
    edges = [(9,3),(9,7),(7,1),(3,1),(7,6),(1,6),(1,5),(6,8),
             (3,0),(4,0),(1,5),(2,0),(10,2),(4,11),(5,11)]
    hierarchy = nx.DiGraph(edges)
    X_train = np.ones((2,len(hierarchy.nodes)))
    y_train = np.array([0,1])
    X_test = np.array([[1,0,1,1,0,0,0,1,0,1,1,0],[1,0,1,1,0,0,0,1,0,1,1,0]])
    relevance = [0.25, 0.23, 0.38, 0.25, 0.28, 0.38, 0.26, 0.31, 0.26, 0.23, 0.21, 0.26]

    return (hierarchy, X_train, y_train, X_test, relevance)


# Test scikit-compatibility
def test_scikit():
    sest = check_estimator(Filter())
    assert sest == None


# Test feature selection of HNB
def test_HNB():
    filter = HNB(hierarchy=small_DAG, k=2)
    filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    assert np.array_equal(filter.get_features(), np.array([[0, 1, 1, 0], [0, 0, 1, 1]]))
    assert np.array_equal(pred, np.array([0, 1]))
    assert filter.get_score(test_y_data, pred) == 0.0


# Test feature selection of HNBs
def test_HNBs():
    filter = HNBs(hierarchy=small_DAG)
    filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    assert np.array_equal(pred, np.array([0, 1]))
    assert np.array_equal(filter.get_features(), np.array([[0, 1, 1, 1], [0, 0, 1, 1]]))
    assert filter.get_score(test_y_data, pred) == 0.0


# Test feature selection of RNB
def test_RNB():
    filter = RNB(hierarchy=small_DAG, k=2)
    filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    assert np.array_equal(pred, np.array([0, 1]))
    assert np.array_equal(filter.get_features(), np.array([[0, 1, 1, 0], [0, 1, 1, 0]]))
    assert filter.get_score(test_y_data, pred) == 0.0

# Test feature selection of MRT
def test_MRT(data1):
    hierarchy, X_train, y_train, X_test, relevance= data1
    filter = MRT(nx.to_numpy_array(hierarchy))
    filter.fit_selector(X_train=X_train, y_train=y_train, X_test=X_test) 
    filter._relevance = relevance
    filter._feature_tree = hierarchy
    filter.select_and_predict(predict=True, saveFeatures=True)
    features = filter.get_features()
    result_features = np.array([[0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0],[0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0]])
    assert(features.all()==result_features.all())

# Test feature selection of HIP
def test_HIP(data1):
    hierarchy, X_train, y_train, X_test, relevance = data1
    filter = HIP(nx.to_numpy_array(hierarchy))
    filter.fit_selector(X_train=X_train, y_train=y_train, X_test=X_test) 
    filter._relevance = relevance
    filter._feature_tree = hierarchy
    filter.select_and_predict(predict=True, saveFeatures=True)
    features = filter.get_features()
    result_features = np.array([[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
    assert(features.all()==result_features.all())