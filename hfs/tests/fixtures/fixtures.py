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
