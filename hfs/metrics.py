import numpy as np
from info_gain.info_gain import info_gain, info_gain_ratio
from numpy.linalg import norm
from pyitlib import discrete_random_variable as drv
from scipy import sparse


def lift(data, labels):
    """returns list including lift value for each feature"""
    lift_values = []
    num_samples, num_features = data.shape

    for index in range(num_features):
        # deal with sparse matrices
        if sparse.issparse(data):
            data = data.tocsr()
            column = data[:, index]
            non_zero_values = column.size
        else:
            column = data[:, index]
            non_zero_values = np.count_nonzero(column)

        prob_feature = non_zero_values / num_samples

        if non_zero_values > 0:
            prob_event_conditional = (
                len(
                    [
                        value
                        for index, value in enumerate(column)
                        if value != 0 and labels[index] != 0
                    ]
                )
                / non_zero_values
            )

            lift_values.append(prob_event_conditional / prob_feature)
        else:
            lift_values.append(0)
    return lift_values


def information_gain(data, labels):
    ig_values = []
    for column_index in range(data.shape[1]):
        ig = info_gain(data[:, column_index], labels)
        ig_values.append(ig)
    return ig_values


def conditional_mutual_information(node1, node2, y):
    return drv.information_mutual_conditional(node1, node2, y)


def cosine_similarity(i: np.ndarray, j: np.ndarray):
    return np.dot(i, j) / (norm(i) * norm(j))


def gain_ratio(data, labels):
    gr_values = []
    for column_index in range(data.shape[1]):
        gr = info_gain_ratio(data[:, column_index], labels)
        gr_values.append(gr)
    return gr_values


def pearson_correlation(x: np.array, y: np.array):
    return np.corrcoef(x, y)[0, 1]
