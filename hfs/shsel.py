import statistics

import numpy as np
from scipy import sparse
from sklearn.utils.validation import check_X_y

from .feature_selection import HierarchicalFeatureSelector
from .helpers import get_paths, information_gain


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
