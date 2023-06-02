"""
Sklearn compatible estimators for preprocessing hierarchical data
"""
import numpy as np
from networkx.algorithms.dag import ancestors
from networkx.algorithms.traversal import bfs_successors
from sklearn.base import BaseEstimator, TransformerMixin, check_array, check_is_fitted

from .base import HierarchicalEstimator
from .helpers import get_leaves


class HierarchicalPreprocessor(HierarchicalEstimator):
    def __init__(self, hierarchy: np.ndarray = None):
        self.hierarchy = hierarchy

    def fit(self, X, y, columns: list[str] = []):
        """Fitting function that sets parameters used to transform the data

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.


        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=True)
        super().fit(X, y, columns)
        self.n_features_ = X.shape[1]

    def transform(self, X):
        """A transform function that updates the dataset so that it conforms to the hierarchy constrictions.
        That means if a feature is 1, all of its ancestors need to be 1. If it is 0, all of its descendants need to be 0.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, "n_features_")

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_:
            raise ValueError(
                "Shape of input is different from what was seen" "in `fit`"
            )
        return self._propagate_ones(X)

    def _propagate_ones(self, X):
        nodes = bfs_successors(self._feature_tree, "ROOT").reverse()
        visited = []
        for node in nodes:
            if node in visited:
                continue
            column_index = self._columns.index(node)
            ancestor_nodes = ancestors(self._feature_tree, "ROOT")
            ancestor_nodes = [node for node in ancestor_nodes if node not in visited]
            for row_index, entry in enumerate(X[:, column_index]):
                if entry == 1:
                    for ancestor in ancestor_nodes:
                        index = self._columns.index(ancestor)
                        X[row_index, index] = 1
                        visited.append(ancestor)

        return X
