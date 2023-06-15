import numpy as np
import pytest

from ..feature_selection import (
    HierarchicalFeatureSelector,
    HillClimbingSelector,
    SHSELSelector,
    TSELSelector,
)
from .fixtures.fixtures import (
    data1,
    data1_2,
    data2,
    data3,
    data_shsel_selection,
    result_hill_selection,
    result_shsel1,
    result_shsel2,
    result_shsel3,
    result_shsel_selection,
    result_tsel1,
    result_tsel2,
    result_tsel3,
    wrong_hierarchy_X,
    wrong_hierarchy_X1,
)


@pytest.mark.parametrize(
    "data, result",
    [
        (data1(), result_tsel1()),
        (data2(), result_tsel2()),
        (data3(), result_tsel3()),
        (data1_2(), result_tsel1()),
    ],
)
def test_TSEL_selection(data, result):
    X, y, hierarchy, columns = data
    expected, support = result
    selector = TSELSelector(hierarchy)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data, result",
    [
        (data1(), result_shsel1()),
        (data2(), result_shsel2()),
        (data3(), result_shsel3()),
        (data1_2(), result_shsel1()),
    ],
)
def test_SHSEL_selection(data, result):
    X, y, hierarchy, columns = data
    expected, support = result
    selector = SHSELSelector(hierarchy)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data, result",
    [
        (data_shsel_selection(), result_shsel_selection()),
        (data1(), result_shsel1()),
        (data1_2(), result_shsel1()),
    ],
)
def test_SHSEL_selection_with_initial_selection(data, result):
    X, y, hierarchy, columns = data
    expected, support = result
    selector = SHSELSelector(hierarchy, similarity_threshold=0.8)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data, result",
    [
        (data1(), result_hill_selection()),
        (data1_2(), result_hill_selection()),
    ],
)
def test_HillClimbing_selection(data, result):
    X, y, hierarchy, columns = data
    expected, support = result
    selector = HillClimbingSelector(hierarchy)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data",
    [wrong_hierarchy_X(), wrong_hierarchy_X1()],
)
def test_HierarchicalFeatureSelector(data):
    X, hierarchy, columns = data
    selector = HierarchicalFeatureSelector(hierarchy)
    with pytest.warns(UserWarning):
        selector.fit(X, columns=columns)
