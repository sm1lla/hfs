import random

import networkx as nx
import numpy as np
import pandas as pd


def data1():
    columns = ["A", "B", "C", "D", "E"]
    df = pd.DataFrame(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        columns=columns,
    )
    edges = [(0, 1), (1, 2), (0, 3), (0, 4)]
    hierarchy = nx.to_numpy_array(nx.DiGraph(edges))
    y = np.array([0, 0, 0, 0, 1])
    X = df.to_numpy()
    return (X, y, hierarchy)


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
    hierarchy = nx.to_numpy_array(nx.DiGraph(edges))
    y = np.array([1, 0, 0, 1, 1])
    return (X, y, hierarchy)


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

    hierarchy = None
    y = np.array([1, 0, 0, 1, 1])
    return (X, y, hierarchy)


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
    return (X, y, hierarchy)


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


def result_hill_selection():
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
