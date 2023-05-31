"""ee
Sklearn compatible estimators for feature selection
"""
import networkx as nx
import numpy as np
from networkx.algorithms.dag import descendants, is_directed_acyclic_graph
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .helpers import create_feature_tree, get_paths, lift


class TreeBasedFeatureSelector(TransformerMixin, BaseEstimator):
    """A tree-based feature selection method for hierarchical features"""

    def __init__(self, hierarchy: nx.DiGraph = None):
        self.hierarchy = hierarchy

    def fit(self, X: np.ndarray, y: np.ndarray, columns: list[str] = []):
        """A reference implementation of a fitting function for a transformer.

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
        self._columns = columns
        X, y = check_X_y(X, y, accept_sparse=True)
        if not self._columns:
            self._set_columns(X.shape[1])
        if not self.hierarchy:
            self._feature_tree = nx.DiGraph()
        else:
            self._feature_tree = self.hierarchy
        self._feature_tree = create_feature_tree(self._feature_tree, self._columns)
        paths = get_paths(self._feature_tree)
        lift_values = lift(X, y)
        self._node_to_lift = {
            self._columns[index]: lift_values[index]
            for index, _ in enumerate(self._columns)
        }
        self.representatives_ = self.find_representatives(paths)
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, "representatives_")
        if X.shape[1] != len(self._columns):
            raise ValueError(
                "Shape of input is different from what was seen" "in `fit`"
            )
        column_indices = [self._columns.index(node) for node in self.representatives_]
        columns = [X[:, index] for index in column_indices]
        return np.column_stack(columns)

    def find_representatives(self, paths: list[list[str]]):
        representatives = set()
        for path in paths:
            path.remove("ROOT")
            max_node = max(path, key=lambda x: self._node_to_lift[x])
            representatives.add(max_node)
        return self._filter_representatives(representatives)

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

    def _set_columns(self, num_columns):
        self._columns = [str(index) for index in range(num_columns)]
