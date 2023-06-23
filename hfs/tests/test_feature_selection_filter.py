import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from hfs.tan import Tan


from ..mr import MRT
from ..hip import HIP
from ..filter import Filter
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

    return (hierarchy, X_train, y_train, X_test, relevance,)

@pytest.fixture
def data2():
    edges = [(4,0),(0,3),(2,3),(5,2),(5,1)]
    hierarchy = nx.DiGraph(edges)
    # to initialize graph, setting the real data later manually as we have to adjust the feature tree before
    # to not cause a validation error
    X_train_ones = np.ones((9,len(hierarchy.nodes)))
    X_train = np.array([[1,1,1,1,1,1],[0,1,0,0,1,1],[1,1,0,0,1,1],
                        [0,1,1,0,1,1],[0,0,1,0,1,1],[0,0,0,0,0,1],
                        [0,0,1,1,0,1],[1,0,0,1,1,0],[0,1,0,0,0,1]])
    y_train = np.array([0,1,1,0,1,1,0,1,1])
    X_test = np.array([[0,0,1,0,1,1],[0,1,1,0,1,1]])
    columns = get_columns_for_numpy_hierarchy(hierarchy, X_train.shape[1])
    sorted_edges = [(5,0),(4,3),(2,4),(2,3),(1,3),(5,2),(5,1),(1,4),(1,2),(1,0),(2,0),(4,0),(5,4),(1,3),(5,3),(4,0)]
    resulted_features = np.array([[0., 1., 1., 1., 1., 0.], [0., 1., 1., 1., 1., 0.]])
    return (hierarchy, X_train_ones, X_train, y_train, X_test, resulted_features)


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

def test_TAN(data2): #mst rename
    hierarchy, X_train_ones, X_train, y_train, X_test, resulted_features = data2
    filter = Tan(nx.to_numpy_array(hierarchy))
    filter.fit_selector(X_train=X_train_ones, y_train=y_train, X_test=X_test) 
    filter._xtrain = X_train
    filter._feature_tree = hierarchy
    filter.select_and_predict(predict=True, saveFeatures=True)
    f = filter.get_features()
    assert(resulted_features.all() == f.all())