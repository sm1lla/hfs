import numpy as np
import pytest
from sklearn.utils.estimator_checks import check_estimator

from hfs.tan import Tan

from ..lazyHierarchicalFeatureSelector import LazyHierarchicalFeatureSelector
from ..hip import HIP
from ..hnb import HNB
from ..hnbs import HNBs
from ..mr import MR
from ..rnb import RNB
from .fixtures.fixtures import *


@pytest.fixture
def data1():
    edges = [
        (9, 3),
        (9, 7),
        (7, 1),
        (3, 1),
        (7, 6),
        (1, 6),
        (1, 5),
        (6, 8),
        (3, 0),
        (4, 0),
        (1, 5),
        (2, 0),
        (10, 2),
        (4, 11),
        (5, 11),
    ]
    hierarchy = nx.DiGraph(edges)
    X_train = np.ones((2, len(hierarchy.nodes)))
    y_train = np.array([0, 1])
    X_test = np.array(
        [[1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0]]
    )
    relevance = [0.25, 0.23, 0.38, 0.25, 0.28, 0.38, 0.26, 0.31, 0.26, 0.23, 0.21, 0.26]

    return (
        hierarchy,
        X_train,
        y_train,
        X_test,
        relevance,
    )


@pytest.fixture
def data2():
    edges = [(4, 0), (0, 3), (2, 3), (5, 2), (5, 1)]
    hierarchy = nx.DiGraph(edges)
    X_train_ones = np.ones((9, len(hierarchy.nodes)))
    X_train = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 1, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 1],
        ]
    )
    y_train = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
    X_test = np.array([[0, 0, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1]])
    resulted_features = np.array(
        [[0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]]
    )
    return (hierarchy, X_train_ones, X_train, y_train, X_test, resulted_features)


@pytest.mark.parametrize(
    "data",
    [
        lazy_data2(),
    ],
)
# Test feature selection of HNB
def test_HNB(data):
    small_DAG, train_x_data, train_y_data, test_x_data, test_y_data = data
    filter = HNB(hierarchy=small_DAG, k=2)
    filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    assert np.array_equal(filter.get_features(), np.array([[0, 1, 1, 0], [0, 0, 1, 1]]))
    assert np.array_equal(pred, np.array([0, 1]))
    assert filter.get_score(test_y_data, pred)["accuracy"] == 0.0  # accuracy
    assert filter.get_score(test_y_data, pred)["1"]["recall"] == 0.0  # sensitivity
    assert filter.get_score(test_y_data, pred)["0"]["recall"] == 0.0  # specivity
    assert filter.get_score(test_y_data, pred)["sensitivityxspecificity"] == 0.0


@pytest.mark.parametrize(
    "data",
    [
        lazy_data2(),
    ],
)
# Test feature selection of HNBs
def test_HNBs(data):
    small_DAG, train_x_data, train_y_data, test_x_data, test_y_data = data
    filter = HNBs(hierarchy=small_DAG)
    filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    assert np.array_equal(pred, np.array([0, 1]))
    assert np.array_equal(filter.get_features(), np.array([[0, 1, 1, 1], [0, 0, 1, 1]]))
    assert filter.get_score(test_y_data, pred)["accuracy"] == 0.0  # accuracy
    assert filter.get_score(test_y_data, pred)["1"]["recall"] == 0.0  # sensitivity
    assert filter.get_score(test_y_data, pred)["0"]["recall"] == 0.0  # specivity
    assert filter.get_score(test_y_data, pred)["sensitivityxspecificity"] == 0.0


@pytest.mark.parametrize(
    "data",
    [
        lazy_data2(),
    ],
)
# Test feature selection of RNB
def test_RNB(data):
    small_DAG, train_x_data, train_y_data, test_x_data, test_y_data = data
    filter = RNB(hierarchy=small_DAG, k=2)
    filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    assert np.array_equal(pred, np.array([0, 1]))
    assert np.array_equal(filter.get_features(), np.array([[0, 1, 1, 0], [0, 1, 1, 0]]))


@pytest.mark.parametrize(
    "data",
    [
        lazy_data1(),
    ],
)
# Test feature selection of MR
def test_MR(data):
    hierarchy, X_train, y_train, X_test, y_test, relevance = data
    filter = MR(nx.to_numpy_array(hierarchy))
    filter.fit_selector(X_train=X_train, y_train=y_train, X_test=X_test)
    filter._relevance = relevance
    filter._hierarchy = hierarchy
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    features = filter.get_features()
    result_features = np.array(
        [[0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0]]
    )
    assert features.all() == result_features.all()
    assert filter.get_score(y_test, pred)["accuracy"] == 0.5  # accuracy
    assert filter.get_score(y_test, pred)["1"]["recall"] == 0.0  # sensitivity
    assert filter.get_score(y_test, pred)["0"]["recall"] == 1.0  # specivity
    assert filter.get_score(y_test, pred)["sensitivityxspecificity"] == 0.0


@pytest.mark.parametrize(
    "data",
    [
        lazy_data1(),
    ],
)
# Test feature selection of HIP
def test_HIP(data):
    hierarchy, X_train, y_train, X_test, y_test, relevance = data
    filter = HIP(nx.to_numpy_array(hierarchy))
    filter.fit_selector(X_train=X_train, y_train=y_train, X_test=X_test)
    filter._relevance = relevance
    filter._hierarchy = hierarchy
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    features = filter.get_features()
    result_features = np.array(
        [[1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
    )
    assert features.all() == result_features.all()
    assert filter.get_score(y_test, pred)["accuracy"] == 0.5  # accuracy
    assert filter.get_score(y_test, pred)["1"]["recall"] == 0.0  # sensitivity
    assert filter.get_score(y_test, pred)["0"]["recall"] == 1.0  # specivity
    assert filter.get_score(y_test, pred)["sensitivityxspecificity"] == 0.0


@pytest.mark.parametrize(
    "data",
    [
        lazy_data3(),
    ],
)
def test_TAN(data):
    hierarchy, X_train_ones, X_train, y_train, X_test, y_test, resulted_features = data
    filter = Tan(nx.to_numpy_array(hierarchy))
    filter.fit_selector(X_train=X_train_ones, y_train=y_train, X_test=X_test)
    filter._xtrain = X_train
    filter._hierarchy = hierarchy
    filter.select_and_predict(predict=True, saveFeatures=True)
    f = filter.get_features()
    assert resulted_features.all() == f.all()
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    assert filter.get_score(y_test, pred)["accuracy"] == 0.5  # accuracy
    assert filter.get_score(y_test, pred)["1"]["recall"] == 1.0  # sensitivity
    assert filter.get_score(y_test, pred)["0"]["recall"] == 0.0  # specivity
    assert filter.get_score(y_test, pred)["sensitivityxspecificity"] == 0.0
