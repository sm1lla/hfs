"""ee
Sklearn compatible estimators for feature selection
"""
import networkx as nx
import numpy as np
from networkx.algorithms.dag import descendants
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .helpers import create_feature_tree, get_paths, lift


# TODO : Use SelectorMixin
class TreeBasedFeatureSelector(TransformerMixin, BaseEstimator):
    """A tree-based feature selection method for hierarchical features"""

    def __init__(
        self, hierarchy: np.ndarray = None, use_original_implementation: bool = True
    ):
        self.hierarchy = hierarchy
        self.use_original_implementation = use_original_implementation

    # TODO : check if columns parameter is really needed and think about how input should look like
    def fit(self, X, y, columns: list[str] = []):
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

        self._columns = columns
        if not self._columns:
            self._columns = range(X.shape[1])

        if self.hierarchy is None:
            self._feature_tree = nx.DiGraph()
        else:
            self._feature_tree = nx.from_numpy_array(
                self.hierarchy, create_using=nx.DiGraph
            )

        # Build feature tree
        self._feature_tree = create_feature_tree(self._feature_tree, self._columns)

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

    def transform(self, X: np.ndarray):
        # Only allow transform after estimator was fitted
        check_is_fitted(self, "representatives_")

        # Input validation
        X = check_array(X, accept_sparse=True)
        if X.shape[1] != len(self._columns):
            raise ValueError(
                "Shape of input is different from what was seen" "in `fit`"
            )

        # Keep selected columns
        column_indices = [self._columns.index(node) for node in self.representatives_]
        columns = [X[:, index] for index in column_indices]
        return np.column_stack(columns)

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
