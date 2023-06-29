import numpy as np
from networkx import ancestors, descendants
from scipy.sparse import issparse
from sklearn.utils.validation import check_X_y

from .feature_selection import HierarchicalFeatureSelector
from .helpers import gain_ratio


class GreedyTopDownSelector(HierarchicalFeatureSelector):
    """Greedy Top Down feature selection method for hierarchical features proposed by Lu et al 2013"""

    def __init__(
        self,
        hierarchy: np.ndarray = None,
    ):
        super().__init__(hierarchy)

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
        if issparse(X):
            X = X.tocsr()
        super().fit(X, y, columns)

        # Feature Selection Algorithm
        self.calculate_heuristic_function(X, y)
        self._fit()

        self.is_fitted_ = True
        return self

    def calculate_heuristic_function(self, X, y):
        gr_values = gain_ratio(X, y)
        self.heuristic_function_values_ = dict(zip(self._columns, gr_values))

    def _fit(self):
        self.representatives_ = []
        top_level_nodes = self._feature_tree.successors("ROOT")
        for node in top_level_nodes:
            branch_nodes = list(descendants(self._feature_tree, node))
            branch_nodes.append(node)
            branch_nodes.sort(
                reverse=True, key=lambda x: self.heuristic_function_values_[x]
            )
            while branch_nodes:
                selected = branch_nodes.pop(0)
                self.representatives_.append(selected)
                remove_nodes = list(descendants(self._feature_tree, selected))
                remove_nodes.extend(list(ancestors(self._feature_tree, selected)))
                if "ROOT" in remove_nodes:
                    remove_nodes.remove("ROOT")
                branch_nodes = [
                    node for node in branch_nodes if node not in remove_nodes
                ]
