import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .helpers import create_feature_tree


class HierarchicalEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, hierarchy: np.ndarray = None):
        self.hierarchy = hierarchy

    def fit(self, X, y, columns: list[str] = []):
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

        self._columns = columns
        self._num_features = X.shape[1]
        if not self._columns:
            self._columns = range(self._num_features)

        if self.hierarchy is None:
            self._feature_tree = nx.DiGraph()
        else:
            self._feature_tree = nx.from_numpy_array(
                self.hierarchy, create_using=nx.DiGraph
            )

        # Build feature tree
        self._feature_tree = create_feature_tree(self._feature_tree, self._columns)

        return self
