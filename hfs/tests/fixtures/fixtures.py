import math
import random

import networkx as nx
import numpy as np
import pandas as pd
from info_gain.info_gain import info_gain, info_gain_ratio

from hfs.helpers import cosine_similarity, get_columns_for_numpy_hierarchy


def data1():
    X = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    edges = [(0, 1), (1, 2), (0, 3), (0, 4)]
    hierarchy = nx.DiGraph(edges)
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    y = np.array([0, 0, 0, 0, 1])
    return (X, y, hierarchy, columns)


def data1_2():
    X = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    edges = [(0, 4), (0, 3), (0, 1), (1, 2)]
    hierarchy = nx.DiGraph(edges)
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    y = np.array([0, 0, 0, 0, 1])
    return (X, y, hierarchy, columns)


def data2():
    X = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
    )
    edges = [(0, 1), (1, 2), (2, 3), (0, 4)]
    hierarchy = nx.DiGraph(edges)
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    y = np.array([1, 0, 0, 1, 1])
    return (X, y, hierarchy, columns)


def data2_1():
    X = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
    )
    edges = [(0, 1), (1, 2), (1, 3)]
    hierarchy = nx.DiGraph(edges)
    hierarchy.add_node(4)
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    y = np.array([1, 0, 0, 1, 1])
    return (X, y, hierarchy, columns)


def data3():
    X = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
    )

    hierarchy = nx.DiGraph()
    hierarchy.add_nodes_from([0, 1, 2, 3, 4])
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    y = np.array([1, 0, 0, 1, 1])
    return (X, y, hierarchy, columns)


def result_tsel1():
    result = np.array([[0], [0], [0], [0], [1]])
    support = np.array([True, False, False, False, False])
    return (result, support)


def result_tsel2():
    result = np.array(
        [
            [1, 1],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 0],
        ]
    )
    support = np.array([False, True, False, False, True])
    return (result, support)


def result_tsel3():
    result = data3()[0]
    support = np.array([True, True, True, True, True])
    return (result, support)


def result_shsel1():
    return result_tsel1()


def result_shsel2():
    result = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 0],
        ],
    )
    support = np.array([False, False, True, False, True])
    return (result, support)


def result_shsel3():
    return result_tsel3()


def data_shsel_selection():
    X = data2()[0]
    y = data2()[1]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    hierarchy = nx.to_numpy_array(nx.DiGraph(edges))
    columns = None
    return (X, y, hierarchy, columns)


def result_shsel_selection():
    result = np.array(
        [
            [0],
            [1],
            [1],
            [0],
            [0],
        ],
    )
    support = np.array([False, False, True, False, False])
    return (result, support)


def result_gtd_selection2():
    result = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 0],
        ],
    )
    support = np.array([False, False, True, False, True])
    return (result, support)


def result_gtd_selection2_1():
    result = np.array(
        [
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
    )
    support = np.array([False, False, True, True, True])
    return (result, support)


def result_hill_selection_td():
    result = pd.DataFrame(
        [
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
    )
    support = np.array([False, True, False, True, True])
    return (result, support)


def result_hill_selection_bu():
    k = 3
    result = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
    )
    support = np.array([False, False, True, True, True])
    return (result, support, k)


def wrong_hierarchy_X():
    X = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    hierarchy = nx.to_numpy_array(nx.DiGraph([(0, 1)]))
    columns = [0, 1, 2]
    return (X, hierarchy, columns)


def wrong_hierarchy_X1():
    X = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    hierarchy = nx.to_numpy_array(nx.DiGraph([(0, 1), (1, 2), (3, 4), (0, 5)]))
    columns = [0, 1, 2]
    return (X, hierarchy, columns)


def result_score_matrix1():
    return np.array(
        [
            [1, 0, 0, 0, 1],
            [2, 0, 0, 1, 1],
            [3, 1, 1, 1, 1],
            [4, 2, 1, 1, 1],
            [5, 2, 1, 1, 1],
        ]
    )


def result_comparison_matrix_td1():
    return np.array(
        [
            [0.0, math.sqrt(2), math.sqrt(7), math.sqrt(15), math.sqrt(22)],
            [math.sqrt(2), 0.0, math.sqrt(3), math.sqrt(9), math.sqrt(14)],
            [math.sqrt(7), math.sqrt(3), 0.0, math.sqrt(2), math.sqrt(5)],
            [math.sqrt(15), math.sqrt(9), math.sqrt(2), 0.0, 1.0],
            [math.sqrt(22), math.sqrt(14), math.sqrt(5), 1.0, 0.0],
        ]
    )


def result_comparison_matrix_bu(matrix: np.ndarray):
    result = np.zeros((5, 5))
    for x in range(5):
        for y in range(5):
            result[x, y] = cosine_similarity(matrix[x, :], matrix[y, :])
    return result


def result_comparison_matrix_bu1():
    matrix = result_score_matrix1()
    return result_comparison_matrix_bu(matrix)


def result_fitness_funtion_td1():
    alpha = 0.99
    doc1 = math.sqrt(22) / (1 + alpha * (math.sqrt(2) + math.sqrt(7) + math.sqrt(15)))
    doc2 = math.sqrt(14) / (1 + alpha * (math.sqrt(2) + math.sqrt(3) + math.sqrt(9)))
    doc3 = math.sqrt(5) / (1 + alpha * (math.sqrt(7) + math.sqrt(3) + math.sqrt(2)))
    doc4 = 1.0 / (1 + alpha * (math.sqrt(15) + math.sqrt(9) + math.sqrt(2)))
    doc5 = (math.sqrt(22) + math.sqrt(14) + math.sqrt(5) + 1.0) / 1.0

    return doc1 + doc2 + doc3 + doc4 + doc5


def result_fitness_funtion_bu1():
    alpha = 3
    n = 5
    beta = 0.01
    k = 3
    selected_nearest_neighbors = [[1, 2], [2, 0], [3, 1], [1, 2], []]
    result = sum([len(x) for x in selected_nearest_neighbors])
    result = result * (1 + beta * (alpha - n) / alpha)
    return (result, k)


def result_score_matrix2():
    return np.array(
        [
            [3, 1, 0, 0, 1],
            [4, 3, 2, 1, 0],
            [3, 2, 1, 0, 0],
            [2, 0, 0, 0, 1],
            [2, 1, 0, 0, 0],
        ]
    )


def result_comparison_matrix_bu2():
    matrix = result_score_matrix2()
    return result_comparison_matrix_bu(matrix)


def result_score_matrix3():
    return np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
    )


def result_comparison_matrix_bu3():
    matrix = result_score_matrix3()
    return result_comparison_matrix_bu(matrix)


def result_gr_values2():
    y = np.array([1, 0, 0, 1, 1])
    return [
        info_gain_ratio(np.array([1, 1, 1, 1, 1]), y),
        info_gain_ratio(np.array([1, 1, 1, 0, 1]), y),
        info_gain_ratio(np.array([0, 1, 1, 0, 0]), y),
        info_gain_ratio(np.array([0, 1, 0, 0, 0]), y),
        info_gain_ratio(np.array([1, 0, 0, 1, 0]), y),
    ]


def result_ig_values2():
    y = np.array([1, 0, 0, 1, 1])
    return [
        info_gain(np.array([1, 1, 1, 1, 1]), y),
        info_gain(np.array([1, 1, 1, 0, 1]), y),
        info_gain(np.array([0, 1, 1, 0, 0]), y),
        info_gain(np.array([0, 1, 0, 0, 0]), y),
        info_gain(np.array([1, 0, 0, 1, 0]), y),
    ]


_feature_number = 9


def getFixedDag():
    return nx.to_numpy_array(
        nx.DiGraph(
            [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (4, 6), (4, 7), (3, 7), (5, 8)]
        )
    )


def rand():
    return random.getrandbits(1)


def randomLinesWithAssertions(y):
    b = rand()
    c = rand()
    d = rand()
    e = rand() if b == 1 else 0
    f = rand() if b == 1 else 0
    g = rand() if e * d == 1 else 0
    h = rand() if e == 1 and d == 1 else 0
    i = rand() if f == 1 else 0
    return (1, b, c, d, e, f, g, h, i)


def getFixedData(instance_number):
    df = pd.DataFrame(columns=[i for i in range(0, _feature_number)])
    y = np.random.randint(0, 2, instance_number)
    for row in range(0, instance_number):
        df.loc[len(df)] = randomLinesWithAssertions(y)
    return df.to_numpy(), y


big_DAG = getFixedDag()
data = getFixedData(20)

small_DAG = nx.to_numpy_array(nx.DiGraph([(0, 1), (0, 2), (1, 2), (1, 3)]))
train_x_data = np.array([[1, 1, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
train_y_data = np.array([0, 0, 1, 1])
test_x_data = np.array([[1, 1, 0, 0], [1, 1, 1, 0]])
test_y_data = np.array([1, 0])
