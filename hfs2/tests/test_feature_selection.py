import sys
sys.path.append('/home/kathrin/hfs/hfs2/')


from sklearn.utils.estimator_checks import check_estimator
from filter import Filter
from hnb import HNB
from hnbs import HNBs
from rnb import RNB
import numpy as np
from .fixtures.fixtures import *
import pytest

# Test scikit-compatibility
def test_scikit():
    sest = check_estimator(Filter())
    assert(sest == None)


# Test feature selection of HNB
def test_HNB():
    filter = HNB(graph_data=small_DAG, k=2)
    filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    assert(np.array_equal(filter.get_features(),np.array([[0, 1, 1, 0], [0, 0, 1, 1]])))
    assert(np.array_equal(pred, np.array([0, 1])))
    assert(filter.get_score(test_y_data, pred) == 0.0)

# Test feature selection of HNBs
def test_HNBs():
    filter = HNBs(graph_data=small_DAG)
    filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    assert(np.array_equal(pred,np.array([0, 1])))
    assert(np.array_equal(filter.get_features(), np.array([[0, 1, 1, 1], [0, 0, 1, 1]])))
    assert(filter.get_score(test_y_data, pred) == 0.0)

# Test feature selection of RNB
def test_RNB():
    filter = RNB(graph_data=small_DAG, k=2)
    filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    assert(np.array_equal(pred,np.array([0, 1])))
    assert(np.array_equal(filter.get_features(), np.array([[0, 1, 1, 0], [0, 1, 1, 0]])))
    assert(filter.get_score(test_y_data, pred) == 0.0)


