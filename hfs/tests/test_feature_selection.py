import numpy as np
import pytest

from ..feature_selection import HierarchicalFeatureSelector
from ..gtd import GreedyTopDownSelector
from ..hill_climbing import BottomUpSelector, TopDownSelector
from ..shsel import SHSELSelector
from ..tsel import TSELSelector
from .fixtures.fixtures import (
    data1,
    data1_2,
    data2,
    data2_1,
    data2_2,
    data3,
    data4,
    data_shsel_selection,
    result_comparison_matrix_bu1,
    result_comparison_matrix_bu2,
    result_comparison_matrix_bu3,
    result_comparison_matrix_td1,
    result_fitness_funtion_bu1,
    result_fitness_funtion_td1,
    result_gtd_selection2,
    result_gtd_selection2_1,
    result_gtd_selection2_2,
    result_hill_selection_bu,
    result_hill_selection_td,
    result_score_matrix1,
    result_score_matrix2,
    result_score_matrix3,
    result_shsel1,
    result_shsel2,
    result_shsel3,
    result_shsel_hfe1,
    result_shsel_hfe2,
    result_shsel_hfe4,
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
        (data1(), result_shsel_hfe1()),
        (data2(), result_shsel_hfe2()),
        (data4(), result_shsel_hfe4()),
    ],
)
def test_leaf_filtering(data, result):
    X, y, hierarchy, columns = data
    expected, support = result
    selector = SHSELSelector(hierarchy, use_hfe_extension=True)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data, result",
    [
        (data1(), result_hill_selection_td()),
        (data1_2(), result_hill_selection_td()),
    ],
)
def test_top_down_selection(data, result):
    X, y, hierarchy, columns = data
    expected, support = result
    selector = TopDownSelector(hierarchy, dataset_type="binary")
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data, result",
    [
        (data1(), result_hill_selection_bu()),
    ],
)
def test_bottom_up_selection(data, result):
    X, y, hierarchy, columns = data
    expected, support, k = result
    selector = BottomUpSelector(hierarchy, k=k, dataset_type="binary")
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
        (data1(), result_comparison_matrix_td1(), TopDownSelector),
        (data1(), result_comparison_matrix_bu1(), BottomUpSelector),
        (data2(), result_comparison_matrix_bu2(), BottomUpSelector),
        (data3(), result_comparison_matrix_bu3(), BottomUpSelector),
    ],
)
def test_comparison_matrix(data, result, Selector):
    X, y, hierarchy, columns = data
    comparison_matrix_expected = result

    selector = Selector(hierarchy)
    selector.fit(X, y, columns)
    comparison_matrix = selector._comparison_matrix(columns)

    assert np.array_equal(comparison_matrix, comparison_matrix_expected)


@pytest.mark.parametrize(
    "data, comparison_matrix, result",
    [
        (
            data1(),
            result_comparison_matrix_bu1(),
            result_fitness_funtion_bu1(),
        ),
    ],
)
def test_calculate_fitness_function_bu(data, comparison_matrix, result):
    X, y, hierarchy, columns = data

    fitness_expected, k = result

    selector = BottomUpSelector(hierarchy, k=k)
    selector.fit(X, y, columns)
    fitness = selector._fitness_function(comparison_matrix)

    assert np.array_equal(fitness, fitness_expected)


@pytest.mark.parametrize(
    "data, comparison_matrix, result",
    [
        (
            data1(),
            result_comparison_matrix_td1(),
            result_fitness_funtion_td1(),
        )
    ],
)
def test_calculate_fitness_function_td(data, comparison_matrix, result):
    X, y, hierarchy, columns = data

    fitness_expected = result

    selector = TopDownSelector(hierarchy)
    selector.fit(X, y, columns)
    fitness = selector._fitness_function(comparison_matrix)

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


@pytest.mark.parametrize(
    "data, result",
    [(data2(), result_gtd_selection2()), (data2_1(), result_gtd_selection2_1())],
)
def test_greedy_top_down_selection(data, result):
    X, y, hierarchy, columns = data
    expected, support = result
    selector = GreedyTopDownSelector(hierarchy)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data, result_redundant, result_not_redundant",
    [(data2_2(), result_gtd_selection2_1(), result_gtd_selection2_2())],
)
def test_greedy_top_down_selection_dag(data, result_redundant, result_not_redundant):
    X, y, hierarchy, columns = data
    expected_redundant, support_redundant = result_redundant
    selector = GreedyTopDownSelector(hierarchy)
    selector.fit(X, y, columns)
    X_transformed = selector.transform(X)
    assert np.array_equal(X_transformed, expected_redundant)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support_redundant)

    expected_not_redundant, support_not_redundant = result_not_redundant
    selector2 = GreedyTopDownSelector(hierarchy, iterate_first_level=False)
    selector2.fit(X, y, columns)
    X_transformed2 = selector2.transform(X)
    assert np.array_equal(X_transformed2, expected_not_redundant)

    support_mask2 = selector2.get_support()
    assert np.array_equal(support_mask2, support_not_redundant)
