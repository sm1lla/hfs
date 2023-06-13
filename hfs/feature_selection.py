"""
Sklearn compatible estimators for feature selection
"""
import math
import statistics

import numpy as np
from networkx.algorithms.dag import descendants
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_array, check_X_y

from .base import HierarchicalEstimator
from .helpers import get_paths, information_gain, lift


class HierarchicalFeatureSelector(SelectorMixin, HierarchicalEstimator):
    def __init__(self, hierarchy: np.ndarray = None):
        super().__init__(hierarchy)

    def fit(self, X, y):
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

        super().fit(X, y)

        self.representatives_ = []

        return self

    def _get_support_mask(self):
        return np.asarray(
            [
                True if index in self.representatives_ else False
                for index in range(self.n_features_)
            ]
        )

    def transform(self, X):
        X = check_array(X, dtype=None, accept_sparse="csr")
        if self.n_features_ != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        return super().transform(X)


class TSELSelector(HierarchicalFeatureSelector):
    """A tree-based feature selection method for hierarchical features proposed by Jeong and Myaeng"""

    def __init__(
        self, hierarchy: np.ndarray = None, use_original_implementation: bool = True
    ):
        super().__init__(hierarchy)
        self.use_original_implementation = use_original_implementation

    # TODO : check if columns parameter is really needed and think about how input should look like
    def fit(self, X, y):
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
        X, y = check_X_y(X, y, accept_sparse=True)

        super().fit(X, y)

        # Feature Selection Algorithm
        paths = get_paths(self._feature_tree)
        lift_values = lift(X, y)
        self._node_to_lift = {
            self._columns[index]: lift_values[index]
            for index, _ in enumerate(self._columns)
        }
        self.representatives_ = self._find_representatives(paths)

        self.is_fitted_ = True
        return self

    def _find_representatives(self, paths: list[list[str]]):
        representatives = set()
        for path in paths:
            path.remove("ROOT")
            max_node = (
                self._select_from_path1(path)
                if self.use_original_implementation
                else self._select_from_path2(path)
            )
            representatives.add(max_node)
        return self._filter_representatives(representatives)

    def _select_from_path1(self, path: list[str]):
        # implementation used in paper by Jeong and Myaeng
        for index, node in enumerate(path):
            if index == len(path) - 1:
                return node
            elif self._node_to_lift[node] >= self._node_to_lift[path[index + 1]]:
                return node

    def _select_from_path2(self, path: list[str]):
        # if multiple nodes are maximum the node closest to the root is returned
        max_node = max(path, key=lambda x: self._node_to_lift[x])
        return max_node

    def _filter_representatives(self, representatives: list[str]):
        updated_representatives = []
        for node in representatives:
            selected_decendents = [
                descendent
                for descendent in descendants(self._feature_tree, node)
                if descendent in representatives
            ]
            if not selected_decendents:
                updated_representatives.append(node)
        return updated_representatives


class SHSELSelector(HierarchicalFeatureSelector):
    """SHSEL feature selection method for hierarchical features proposed by Ristoski and Paulheim"""

    def __init__(
        self,
        hierarchy: np.ndarray = None,
        relevance_metric: str = "IG",
        similarity_threshold=0.99,
    ):
        super().__init__(hierarchy)
        self.relevance_metric = relevance_metric
        self.similarity_threshold = similarity_threshold

    def fit(self, X, y):
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

        super().fit(X, y)

        # Feature Selection Algorithm
        self._calculate_relevance(X, y)
        self._fit()

        self.is_fitted_ = True
        return self

    def _fit(self):
        paths = get_paths(self._feature_tree, reverse=True)
        self._inital_selection(paths)
        self._pruning(paths)

    def _inital_selection(self, paths):
        remove_nodes = set()

        for path in paths:
            for index, node in enumerate(path):
                parent_node = path[index + 1]
                if parent_node == "ROOT":
                    break
                relevance_similarity = 1 - abs(
                    self._relevance_values[parent_node] - self._relevance_values[node]
                )
                if relevance_similarity >= self.similarity_threshold:
                    remove_nodes.add(node)

        self.representatives_ = [
            feature for feature in self._columns if feature not in remove_nodes
        ]

    def _pruning(self, paths):
        paths = get_paths(self._feature_tree, reverse=True)
        updated_representatives = []

        for path in paths:
            path.remove("ROOT")
            average_relevance = statistics.mean(
                [
                    self._relevance_values[node]
                    for node in path
                    if node in self.representatives_
                ]
            )
            for node in path:
                if node in self.representatives_ and round(
                    self._relevance_values[node], 6
                ) >= round(average_relevance, 6):
                    updated_representatives.append(node)

        self.representatives_ = updated_representatives

    def _calculate_relevance(self, X, y):
        if self.relevance_metric == "IG":
            values = information_gain(X, y)
            self._relevance_values = dict(zip(self._columns, values))


class HillClimbingSelector(HierarchicalFeatureSelector):
    """Hill climbing feature selection method for hierarchical features proposed by Wang et al."""

    def __init__(self, hierarchy: np.ndarray = None, alpha: float = 0.99):
        super().__init__(hierarchy)
        self.alpha = alpha

    def fit(self, X, y):
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

        super().fit(X, y)

        # Feature Selection Algorithm
        self.y_ = y
        self._calculate_distances(X)
        self.representatives_ = self._hill_climb_top_down()

        return self

    def _calculate_normalized_frequencies(self, X):
        self.frequency_matrix_ = X
        num_rows, num_columns = X.shape
        for row in range(num_rows):
            max_frequency = max(self.frequency_matrix_[row, :])
            for column in range(num_columns):
                frequency = self.frequency_matrix_[row, column]
                if frequency != 0:
                    self.frequency_matrix_[row, column] = (
                        math.log(1 + (frequency / max_frequency)) + 1
                    )

    def _calculate_distance(self, sample_i, sample_j):
        distance = 0
        for column in range(self.frequency_matrix_.shape[1]):
            difference = (
                self.frequency_matrix_[sample_i, column]
                - self.frequency_matrix_[sample_j, column]
            )
            distance += math.pow(difference, 2)
        return math.sqrt(distance)

    def _calculate_distances(self, X):
        self._calculate_normalized_frequencies(X)
        num_rows = X.shape[0]
        self.distances_ = np.zeros((num_rows, num_rows), dtype=int)
        for row in range(num_rows):
            for column in range(num_rows):
                if self.distances_[row, column] == 0:
                    self.distances_[row, column] = self._calculate_distance(row, column)
                    self.distances_[column, row] = self.distances_[row, column]

    def _fitness_function(self, features: set[int]) -> float:
        result = 0
        for feature in features:
            same_class = [
                sample for sample in features if self.y_[sample] == self.y_[feature]
            ]
            other_class = [sample for sample in features if sample not in same_class]

            nominator = sum(
                [self.distances_[sample, feature] for sample in other_class]
            )
            denominator = 1 + self.alpha * sum(
                [self.distances_[sample, feature] for sample in same_class]
            )
            result += nominator / denominator

        return result

    def _hill_climb_top_down(self) -> list[int]:
        optimal_feature_set = set(self._feature_tree.successors("ROOT"))
        fitness = 0
        best_fitness = 0
        best_feature_set = None

        while True:
            for node in optimal_feature_set:
                children = list(self._feature_tree[node])
                if children:
                    temporary_feature_set = optimal_feature_set
                    temporary_feature_set.remove(node)
                    temporary_feature_set.update(children)
                    temporary_fitness = self._fitness_function(temporary_feature_set)
                    if (temporary_fitness) > best_fitness:
                        best_fitness = temporary_fitness
                        best_feature_set = temporary_feature_set

            if best_fitness > fitness:
                optimal_feature_set = best_feature_set
                fitness = best_fitness
            else:
                break
        return list[optimal_feature_set]
