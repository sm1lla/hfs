import numpy as np
import pytest

from ..feature_selection import HierarchicalFeatureSelector
from ..hill_climbing import TopDownSelector
from ..shsel import SHSELSelector
from ..tsel import TSELSelector
from .fixtures.fixtures import (
    data1,
    data1_2,
    data2,
    data3,
    data_shsel_selection,
    result_distance_matrix_bu1,
    result_distance_matrix_bu2,
    result_distance_matrix_bu3,
    result_distance_matrix_td1,
    result_fitness_funtion_bu1,
    result_fitness_funtion_td1,
    result_hill_selection_bu,
    result_hill_selection_td,
    result_score_matrix1,
    result_score_matrix2,
    result_score_matrix3,
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
    "data, result, Selector",
    [
        (data1(), result_hill_selection_td(), TopDownSelector),
        (data1(), result_hill_selection_bu(), TopDownSelector),
        (data1_2(), result_hill_selection_td(), TopDownSelector),
    ],
)
def test_top_down_selection(data, result, Selector):
    X, y, hierarchy, columns = data
    expected, support = result
    selector = Selector(hierarchy, dataset_type="binary")
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data, result",
    [
        (data1(), result_score_matrix1()),
        (data2(), result_score_matrix2()),
        (data3(), result_score_matrix3()),
    ],
)
def test_calculate_scores(data, result):
    X, y, hierarchy, columns = data
    score_matrix_expected = result

    selector = TopDownSelector(hierarchy, dataset_type="binary")
    selector.fit(X, y, columns)
    score_matrix = selector._calculate_scores(X)

    assert np.array_equal(score_matrix, score_matrix_expected)


@pytest.mark.parametrize(
    "data, result, Selector",
    [
        (data1(), result_distance_matrix_td1(), TopDownSelector),
        (data1(), result_distance_matrix_bu1(), TopDownSelector),
    ],
)
def test_calculate_distances(data, result, Selector):
    X, y, hierarchy, columns = data
    distance_matrix_expected = result

    selector = Selector(hierarchy)
    selector.fit(X, y, columns)
    distance_matrix = selector._calculate_distances(columns)

    assert np.array_equal(distance_matrix, distance_matrix_expected)


@pytest.mark.parametrize(
    "data, distance_matrix, result, Selector",
    [
        (
            data1(),
            result_distance_matrix_td1(),
            result_fitness_funtion_td1(),
            TopDownSelector,
        ),
        (
            data1(),
            result_distance_matrix_bu1(),
            result_fitness_funtion_bu1(),
            TopDownSelector,
        ),
    ],
)
def test_calculate__fitness_function(data, distance_matrix, result, Selector):
    X, y, hierarchy, columns = data

    fitness_expected = result

    selector = Selector(hierarchy)
    selector.fit(X, y, columns)
    fitness = selector._fitness_function(distance_matrix)

    assert np.array_equal(fitness, fitness_expected)


@pytest.mark.parametrize(
    "data",
    [wrong_hierarchy_X(), wrong_hierarchy_X1()],
)
def test_HierarchicalFeatureSelector(data):
    X, hierarchy, columns = data
    selector = HierarchicalFeatureSelector(hierarchy)
    with pytest.warns(UserWarning):
        selector.fit(X, columns=columns)
