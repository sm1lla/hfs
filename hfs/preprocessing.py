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

    def fit(self, X, y=None, columns=None):
        """Fitting function that sets parameters used to transform the data

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        columns:
            need to map features from X to nodes on graph


        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=True)
        super().fit(X, y, columns)
        self._extend_dag(X)
        self._shrink_dag()
        self._find_missing_columns()
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

    def _extend_dag(self, X):
        """Add features that are not represented yet as nodes to the root
        X: dataset with more features than hierarchy
        columns: only the columns that are represented in hierarchy. Each position at column represents the position in X, the value is the number of the node.
        If a position in X is not represented in the hierarchy the value should be set to -1.
        """
        max = len(self._feature_tree.nodes)
        for x in range(len(self._columns)):
            if self._columns[x] == -1:
                if x in self._feature_tree.nodes:
                    self._feature_tree.add_edge("ROOT", max)
                    self._columns[x] = max
                    max += 1
                else:
                    self._feature_tree.add_edge("ROOT", x)
                    self._columns[x] = x

    def _shrink_dag(self):
        leaves = get_irrelevant_leaves(
            x_identifier=self._columns, digraph=self._feature_tree
        )
        while leaves:
            for x in leaves:
                self._feature_tree.remove_node(x)
            leaves = get_irrelevant_leaves(
                x_identifier=self._columns, digraph=self._feature_tree
            )

    def _find_missing_columns(self):
        missing_nodes = [
            node
            for node in self._feature_tree.nodes
            if node not in self._columns and node != "ROOT"
        ]
        self._columns.extend(missing_nodes)

    def _add_columns(self, X):
        X_ = X
        num_rows, num_columns = X.shape
        if num_columns != len(self._columns):
            missing_nodes_indices = list(range(num_columns, len(self._columns)))
            for _ in missing_nodes_indices:
                X_ = np.concatenate([X_, np.zeros((num_rows, 1), dtype=int)], axis=1)
        return X_

    def _propagate_ones(self, X):
        nodes = list(self._feature_tree.nodes)
        nodes.remove("ROOT")

        for node in nodes:
            column_index = self._column_index(node)
            ancestor_nodes = ancestors(self._feature_tree, node)
            ancestor_nodes.remove("ROOT")
            for row_index, entry in enumerate(X[:, column_index]):
                if entry == 1.0:
                    for ancestor in ancestor_nodes:
                        index = self._column_index(ancestor)
                        X[row_index, index] = 1.0

        return X

    def get_hierarchy(self):
        if self.is_fitted_:
            output_hierarchy = self._feature_tree
            output_hierarchy.remove_node("ROOT")
            return networkx.to_numpy_array(self._feature_tree)
        else:
            raise RuntimeError(f"Instance has not been fitted.")
