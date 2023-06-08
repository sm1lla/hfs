"""
Sklearn compatible estimators for feature selection
"""
import statistics

import networkx as nx
import numpy as np
from networkx.algorithms.dag import descendants
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_X_y

from .base import HierarchicalEstimator
from .helpers import get_paths, information_gain, lift


class HierarchicalFeatureSelector(HierarchicalEstimator, SelectorMixin):
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
                for index in range(self._num_features)
            ]
        )


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
