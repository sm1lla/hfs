"""
Sklearn compatible estimators for preprocessing hierarchical data
"""
import networkx
import numpy as np
from networkx.algorithms.dag import ancestors
from networkx.algorithms.traversal import bfs_successors
from sklearn.utils.validation import check_array, check_is_fitted

from .base import HierarchicalEstimator
from .helpers import get_irrelevant_leaves


class HierarchicalPreprocessor(HierarchicalEstimator):
    def __init__(self, hierarchy: np.ndarray = None):
        self.hierarchy = hierarchy

    def fit(self, X, y=None, column_names=None):
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
        super().fit(X, y)
        self._find_missing_columns()
        self._shrink_dag(column_names)
        self.is_fitted_ = True
        return self

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

        X_ = self._add_columns(X)
        X_ = self._propagate_ones(X_)
        return X_

    def _shrink_dag(self, x_identifier):
        leaves = get_irrelevant_leaves(
            x_identifier=x_identifier, digraph=self._feature_tree
        )
        while leaves:
            for x in leaves:
                self._feature_tree.remove_node(x)
            leaves = get_irrelevant_leaves(
                x_identifier=x_identifier, digraph=self._feature_tree
            )

    def _find_missing_columns(self):
        num_nodes = len(self._feature_tree.nodes) - 1
        num_columns = len(self._columns)
        if num_nodes > num_columns:
            missing_nodes = list(range(num_columns, num_nodes))
            self._columns.extend(missing_nodes)

    def _add_columns(self, X):
        X_ = X
        num_rows, num_columns = X.shape
        if num_columns != len(self._columns):
            missing_nodes = list(range(num_columns, len(self._columns)))
            for _ in missing_nodes:
                X_ = np.concatenate([X_, np.zeros((num_rows, 1), dtype=int)], axis=1)
        return X_

    def _propagate_ones(self, X):
        nodes = list(self._feature_tree.nodes)
        nodes.remove("ROOT")

        for node in nodes:
            column_index = self._columns.index(node)
            ancestor_nodes = ancestors(self._feature_tree, node)
            ancestor_nodes.remove("ROOT")
            for row_index, entry in enumerate(X[:, column_index]):
                if entry == 1.0:
                    for ancestor in ancestor_nodes:
                        index = self._columns.index(ancestor)
                        X[row_index, index] = 1.0

        return X

    def get_hierarchy(self):
        if self.is_fitted_:
            return networkx.to_numpy_array(self._feature_tree)
        else:
            raise RuntimeError(f"Instance has not been fitted.")
