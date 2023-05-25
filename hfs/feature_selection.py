"""
Sklearn compatible estimators for feature selection
"""
import networkx as nx
import numpy as np
from networkx.algorithms.dag import descendants, is_directed_acyclic_graph
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y

from .helpers import create_feature_tree, get_paths, lift


class TreeBasedFeatureSelector(TransformerMixin, BaseEstimator):
    """A tree-based feature selection method for hierarchical features"""

    def __init__(self, hierarchy: nx.DiGraph = None, columns: list[str] = []):
        if not hierarchy:
            hierarchy = nx.DiGraph()
        # TODO make sure all labels are in the hierarchy
        self.feature_tree = create_feature_tree(hierarchy)
        self.columns = columns
        self.representatives = None

    def fit(self, X: np.ndarray, y: np.ndarray):
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
        X, y = check_X_y(X, y)

        paths = get_paths(self.feature_tree)
        lift_values = lift(X, y)
        self.node_to_lift = {
            self.columns[index]: lift_values[index]
            for index, _ in enumerate(self.columns)
        }
        self.representatives = self.find_representatives(paths)

        return self

    def transform(self, X: np.ndarray):
        column_indices = [self.columns.index(node) for node in self.representatives]
        columns = [X[:, index] for index in column_indices]
        return np.column_stack(columns)

    def find_representatives(self, paths: list[list[str]]):
        representatives = set()
        for path in paths:
            path.remove("ROOT")
            max_node = max(path, key=lambda x: self.node_to_lift[x])
            representatives.add(max_node)
        return self._filter_representatives(representatives)

    def _filter_representatives(self, representatives: list[str]):
        updated_representatives = []
        for node in representatives:
            selected_decendents = [
                descendent
                for descendent in descendants(self.feature_tree, node)
                if descendent in representatives
            ]
            if not selected_decendents:
                updated_representatives.append(node)
        return updated_representatives
