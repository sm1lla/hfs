import math

import numpy as np
from networkx.algorithms.dag import descendants
from scipy import sparse
from sklearn.utils.validation import check_X_y

from hfs.feature_selection import EagerHierarchicalFeatureSelector
from hfs.helpers import compute_aggregated_values, get_leaves, normalize_score
from hfs.metrics import cosine_similarity


class HillClimbingSelector(EagerHierarchicalFeatureSelector):
    def __init__(
        self,
        hierarchy: np.ndarray = None,
        alpha: float = 0.99,
        dataset_type: str = "binary",
    ):
        super().__init__(hierarchy)
        self.alpha = alpha
        self.dataset_type = dataset_type

    def fit(self, X, y, columns=None):
        """Fitting function that sets self.representatives_ to include the columns that are kept.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int, that should either be 1 or 0.
        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X, y = check_X_y(X, y, accept_sparse=True)

        super().fit(X, y, columns)
        if sparse.issparse(X):
            X = X.tocsr()

        # Feature Selection Algorithm
        self.y_ = y
        self._num_rows = X.shape[0]
        self.representatives_ = self._hill_climb(X)

        return self

    def _hill_climb(self, X):
        raise NotImplementedError

    def _calculate_scores(self, X):
        """Calculate sums of datapoints in X and their children. If the dataset is of the type "numerical" the
        sums are normalized."""
        score_matrix = compute_aggregated_values(
            X.copy(), self._feature_tree, self._columns
        )
        if self.dataset_type == "numerical":
            normalized_matrix = np.zeros_like(score_matrix, dtype=float)
            for row_index in range(self._num_rows):
                for column_index in range(self.n_features_):
                    if self.dataset_type == "numerical":
                        score = score_matrix[row_index, column_index]
                        normalized_matrix[row_index, column_index] = normalize_score(
                            score, max(score_matrix[row_index, :])
                        )
            score_matrix = normalized_matrix
        return score_matrix

    def _compare(
        self,
        sample_i: int,
        sample_j: int,
        feature_set: list[int],
    ):
        raise NotImplementedError

    def _comparison_matrix(self, feature_set: list[int]):
        distances = np.zeros((self._num_rows, self._num_rows), dtype=float)
        for row in range(self._num_rows):
            for column in range(self._num_rows):
                if distances[row, column] == 0:
                    distances[row, column] = self._compare(row, column, feature_set)
                    distances[column, row] = distances[row, column]
        return distances

    def _fitness_function(self, comparison_matrix: np.ndarray) -> float:
        raise NotImplementedError


class TopDownSelector(HillClimbingSelector):
    """Hill climbing feature selection method for hierarchical features proposed by Wang et al.2002"""

    def __init__(
        self,
        hierarchy: np.ndarray = None,
        alpha: float = 0.99,
        dataset_type: str = "binary",
    ):
        super().__init__(hierarchy, alpha=alpha, dataset_type=dataset_type)

    def _hill_climb(self, X) -> list[int]:
        self._score_matrix = self._calculate_scores(X)
        optimal_feature_set = set(self._feature_tree.successors("ROOT"))
        fitness = 0
        best_fitness = 0
        best_feature_set = None

        while True:
            for node in optimal_feature_set:
                children = list(self._feature_tree.successors(node))
                if children:
                    temporary_feature_set = optimal_feature_set.copy()
                    temporary_feature_set.remove(node)
                    temporary_feature_set.update(children)
                    distances = self._comparison_matrix(temporary_feature_set)
                    temporary_fitness = self._fitness_function(distances)
                    if (temporary_fitness) > best_fitness:
                        best_fitness = temporary_fitness
                        best_feature_set = temporary_feature_set

            if best_fitness > fitness:
                optimal_feature_set = best_feature_set
                fitness = best_fitness
            else:
                break
        return list(optimal_feature_set)

    def _compare(
        self,
        sample_i: int,
        sample_j: int,
        feature_set: list[int],
    ):
        return self._calculate_distance(sample_i, sample_j, feature_set)

    def _calculate_distance(
        self,
        sample_i: int,
        sample_j: int,
        feature_set: list[int],
    ):
        distance = 0
        for column in feature_set:
            column_index = self._column_index(column)
            difference = (
                self._score_matrix[sample_i, column_index]
                - self._score_matrix[sample_j, column_index]
            )
            distance += math.pow(difference, 2)
        return math.sqrt(distance)

    def _fitness_function(self, comparison_matrix: np.ndarray) -> float:
        result = 0
        row_indices = range(self._num_rows)
        for row_index in row_indices:
            same_class = [
                sample
                for sample in row_indices
                if self.y_[sample] == self.y_[row_index]
            ]
            other_class = [sample for sample in row_indices if sample not in same_class]

            nominator = sum(
                [comparison_matrix[sample, row_index] for sample in other_class]
            )
            denominator = 1 + self.alpha * sum(
                [comparison_matrix[sample, row_index] for sample in same_class]
            )
            result += nominator / denominator

        return result


class BottomUpSelector(HillClimbingSelector):
    """Hill climbing feature selection method for hierarchical features proposed by Wang et al. 2003"""

    def __init__(
        self,
        hierarchy: np.ndarray = None,
        alpha: float = 0.01,
        k: int = 5,
        dataset_type: str = "binary",
    ):
        # alpha is beta from algorithm in the paper by Wang et al.
        super().__init__(hierarchy, alpha=alpha, dataset_type=dataset_type)
        self.k = k

    def _hill_climb(self, X) -> list[int]:
        self._score_matrix = self._calculate_scores(X)

        current_feature_set = get_leaves(self._feature_tree)
        if current_feature_set == ["ROOT"] or current_feature_set == []:
            return []
        current_fitness = self._fitness_function(
            self._comparison_matrix(current_feature_set)
        )

        unvisited = set(current_feature_set)

        while unvisited:
            temporary_feature_set = current_feature_set.copy()
            node = unvisited.pop()
            parent = list(self._feature_tree.predecessors(node))[
                0
            ]  # does not work with DAG
            if parent != "ROOT":
                temporary_feature_set.append(parent)
                children = list(self._feature_tree.successors(parent))
                updated_feature_set = [
                    node for node in temporary_feature_set if node not in children
                ]
                temporary_fitness = self._fitness_function(
                    self._comparison_matrix(updated_feature_set)
                )
                if temporary_fitness < current_fitness:
                    current_feature_set = temporary_feature_set
                    current_fitness = temporary_fitness
                    unvisited = set(current_feature_set)

        return current_feature_set

    def _compare(
        self,
        sample_i: int,
        sample_j: int,
        feature_set: list[int],
    ):
        return self._calculate_similarity(sample_i, sample_j, feature_set)

    def _calculate_similarity(
        self, sample_i: int, sample_j: int, feature_set: list[int]
    ):
        if "ROOT" in feature_set:
            feature_set = feature_set.remove("ROOT")
        row_i = self._score_matrix[sample_i, feature_set]
        row_j = self._score_matrix[sample_j, feature_set]
        return cosine_similarity(row_i.flatten(), row_j.flatten())

    def _fitness_function(self, comparison_matrix: np.ndarray) -> float:
        number_of_leaf_nodes = len(get_leaves(self._feature_tree))  # alpha from paper
        if number_of_leaf_nodes == 0:
            number_of_leaf_nodes = 1

        threshold_index = self._num_rows - self.k - 1
        k_nearest_neigbors = [
            list(
                np.argpartition(comparison_matrix[row, :], threshold_index)[
                    threshold_index:
                ]
            )
            for row in range(self._num_rows)
        ]

        count = 0
        for row in range(self._num_rows):
            for neighbor in k_nearest_neigbors[row]:
                if self.y_[neighbor] == self.y_[row] and neighbor != row:
                    count += 1

        result = count * (
            1
            + self.alpha
            * (number_of_leaf_nodes - self.n_features_)
            / number_of_leaf_nodes
        )
        return result
