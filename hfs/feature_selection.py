"""
Sklearn compatible estimators for feature selection
"""
import warnings

import numpy as np
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_array

from .base import HierarchicalEstimator


class HierarchicalFeatureSelector(SelectorMixin, HierarchicalEstimator):
    def __init__(self, hierarchy: np.ndarray = None):
        super().__init__(hierarchy)

    def fit(self, X, y=None, columns=None):
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

        super().fit(X, y, columns)
        self._check_hierarchy_X()

        # representatives_ includes all node names for selected nodes, columns maps them to the respective column in X
        self.representatives_ = []

        return self

    def _get_support_mask(self):
        representatives_indices = [
            self._column_index(node) for node in self.representatives_
        ]
        return np.asarray(
            [
                True if index in representatives_indices else False
                for index in range(self.n_features_)
            ]
        )

    def transform(self, X):
        X = check_array(X, dtype=None, accept_sparse="csr")
        if self.n_features_ != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        return super().transform(X)

    def _check_hierarchy_X(self):
        not_in_hierarchy = [
            feature_index
            for feature_index in range(self.n_features_)
            if feature_index not in self._columns
        ]
        if not_in_hierarchy:
            warnings.warn(
                """All columns in X need to be mapped to a node in self.feature_tree. 
            If columns=None the corresponding node's name is the same as the columns index in the dataset. Otherwise it is the node's is in self.columns
            at the index of the column's index"""
            )

        not_in_dataset = [
            node for node in self._feature_tree.nodes() if node not in self._columns
        ]
        if not_in_dataset:
            warnings.warn(
                """The hierarchy should not include any nodes
            that are not mapped to a column in the dataset by the columns parameter"""
            )
