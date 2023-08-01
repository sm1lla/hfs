import statistics

import numpy as np
from scipy import sparse
from sklearn.utils.validation import check_X_y

from .feature_selection import EagerHierarchicalFeatureSelector
from .helpers import (
    compute_aggregated_values,
    get_leaves,
    get_paths,
    information_gain,
    pearson_correlation,
)


class SHSELSelector(EagerHierarchicalFeatureSelector):
    """SHSEL feature selection method for hierarchical features proposed by Ristoski and Paulheim"""

    def __init__(
        self,
        hierarchy: np.ndarray = None,
        relevance_metric: str = "IG",
        similarity_threshold=0.99,
        use_hfe_extension=False,
        preprocess_numerical_data=False,
    ):
        super().__init__(hierarchy)
        self.relevance_metric = relevance_metric
        self.similarity_threshold = similarity_threshold
        self.use_hfe_extension = use_hfe_extension
        self.preprocess_numerical_data = preprocess_numerical_data

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
        if sparse.issparse(X):
            X = X.tocsr()
        super().fit(X, y, columns)

        # Feature Selection Algorithm
        self._calculate_relevance(X, y)
        self._fit(X)

        self.is_fitted_ = True
        return self

    def _fit(self, X):
        if self.preprocess_numerical_data:
            X = self._preprocess(X)
        paths = get_paths(self._feature_tree, reverse=True)
        self._inital_selection(paths, X)
        self._pruning(paths)
        if self.use_hfe_extension:
            self._leaf_filtering()

    def _inital_selection(self, paths, X):
        remove_nodes = set()

        for path in paths:
            for index, node in enumerate(path):
                parent_node = path[index + 1]
                if parent_node == "ROOT":
                    break
                if self.relevance_metric == "IG":
                    similarity = 1 - abs(
                        self._relevance_values[parent_node]
                        - self._relevance_values[node]
                    )
                else:
                    similarity = pearson_correlation(
                        X[:, self._columns.index(parent_node)],
                        X[:, self._columns.index(node)],
                    )
                if similarity >= self.similarity_threshold:
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
        values = information_gain(X, y)
        self._relevance_values = dict(zip(self._columns, values))

    def _preprocess(self, X):
        return compute_aggregated_values("ROOT", X, self._feature_tree, self._columns)

    def _leaf_filtering(self):
        average_ig = statistics.mean(
            [self._relevance_values[node] for node in self.representatives_]
        )

        leaves = self._select_leaves()

        remove_nodes = [
            leaf
            for leaf in leaves
            if self._relevance_values[leaf] < average_ig
            or self._relevance_values[leaf] == 0
        ]
        updated_representatives = [
            node for node in self.representatives_ if node not in remove_nodes
        ]
        self.representatives_ = updated_representatives

    def _select_leaves(self):
        leaves = [
            leaf
            for leaf in get_leaves(self._feature_tree)
            if leaf in self.representatives_
        ]

        paths = get_paths(self._feature_tree)
        max_path_len = max([len(path) for path in paths])
        selected_leaves = []
        for leaf in leaves:
            for path in paths:
                if leaf in path and len(path) != max_path_len:
                    selected_leaves.append(leaf)
        return selected_leaves
