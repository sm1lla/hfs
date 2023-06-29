import numpy as np
from scipy.sparse import issparse
from sklearn.utils.validation import check_X_y

from .feature_selection import HierarchicalFeatureSelector
from .helpers import information_gain


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
        # TODO

        self.is_fitted_ = True
        return self
