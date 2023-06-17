import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_X_y

from .helpers import create_feature_tree


class HierarchicalEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, hierarchy: np.ndarray = None):
        self.hierarchy = hierarchy

    def fit(self, X, y=None, columns=None):
        """Fitting function that creates a DiGraph with a new root node for the hierarchy and initializes the _columns parameter.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or None

        Returns
        -------
        self : object
            Returns self.
        """

        X = check_array(X, accept_sparse=True)

        self.n_features_ = X.shape[1]
        if columns:
            assert len(columns) == self.n_features_
            self._columns = columns
        else:
            self._columns = list(range(self.n_features_))

        self._set_feature_tree()

        return self

    def _set_feature_tree(self):
        if self.hierarchy is None:
            self._feature_tree = nx.DiGraph()
        else:
            self._feature_tree = nx.from_numpy_array(
                self.hierarchy, create_using=nx.DiGraph
            )

        # Build feature tree
        self._feature_tree = create_feature_tree(self._feature_tree)

    def _column_index(self, node):
        return self._columns.index(node)

    def get_columns(self):
        return self._columns

    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        X = check_array(X, dtype=None, accept_sparse="csr")

        if self.n_features_ != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")

        return X
