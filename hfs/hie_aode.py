from collections import defaultdict

import networkx as nx
import numpy as np
from sklearn.naive_bayes import BernoulliNB

from .lazyHierarchicalFeatureSelector import LazyHierarchicalFeatureSelector


class HieAODE(LazyHierarchicalFeatureSelector):
    """
    Select non-redundant features following the algorithm proposed by Wan and Freitas.
    """

    def __init__(self, hierarchy=None):
        """Initializes a HieAODE-Selector.

        Parameters
        ----------
        hierarchy : np.ndarray
            The hierarchy graph as an adjacency matrix.
        """
        self.cpts = dict()
        super(HieAODE, self).__init__(hierarchy)

    def fit_selector(self, X_train, y_train, X_test, columns=None):
        """
        P (y, x_i )
        class_prior

        self.n_ancestors = self.n_descendants = self.n_features
        P (x_k|y)
        ascendants_class_cpt = (self.n_ancestors, self.n_classes, self.n_features, n_values)

        P (x_j|y, x_i)
        feature_descendants_class_cpt = (self.n_features, self.n_descendants, self.n_classes, n_values)
        """
        super(HieAODE, self).fit_selector(X_train, y_train, X_test, columns)
        self.cpts = dict(
            prior=np.full((self.n_features, self.n_classes, 2), -1),  # P(y, x_i )
            descendants=np.full((self.n_features, self.n_features, self.n_classes, 2), -1),  # P(x_j|y, x_i)
            ascendants=np.full((self.n_features, self.n_classes, 2), -1)  # P(x_k|y)
        )

    def select_and_predict(
        self, predict=True, saveFeatures=False, estimator=BernoulliNB()
    ):
        """
        Select features lazy for each test instance and optionally predict target value of test instances
        using the HieAODE algorithm by Wan and Freitas

        Parameters
        ----------
        predict : bool
            true if predictions shall be obtained.
        saveFeatures : bool
            true if features selected for each test instance shall be saved.
        estimator : sklearn-compatible estimator
            Estimator to use for predictions.


        Returns
        -------
        predictions for test input samples, if predict = false, returns empty array.
        """

        for sample_idx in range(self._xtest.shape[0]):
            sample = self._xtest[sample_idx]
            for feature_idx in range(len(sample)):
                self.calculate_class_prior(sample=sample, feature_idx=feature_idx)

                ascendants = nx.ancestors(self._hierarchy, feature_idx)
                for a in range(len(ascendants)):
                    self.calculate_prob_given_ascendant_class(ascendant=a)

                descendants = nx.descendants(self._hierarchy, feature_idx)
                # Remove elements from descendants that are in ascendants
                descendants = [item for item in descendants if item not in ascendants]
                for descendant_idx in range(len(descendants)):
                    self.calculate_prob_descendant_given_class_feature(descendant_idx=descendant_idx, feature_idx=feature_idx)
    def calculate_class_prior(self, sample, feature_idx):
        for c in range(self.n_classes):
            for value in range(2):
                self.cpts["prior"][feature_idx][c][value] = (
                        (np.sum(self._ytrain == c) & (sample == value))
                        / self._ytrain.shape[0]
                )

    def calculate_prob_given_ascendant_class(self, ascendant):
        # Calculate P(x_k | y) where x_k=ascendant and y = c
        for c in range(self.n_classes):
            for value in range(2):
                p_class_ascendant = np.sum(
                    (self._ytrain == c) & (self._xtrain[:, ascendant] == value)
                )
                p_class = np.sum(self._ytrain == c)
                self.cpts["ascendants"][ascendant][c][value] = p_class_ascendant / p_class

    def calculate_prob_descendant_given_class_feature(self, descendant_idx, feature_idx):
        for c in range(self.n_classes):
            for value in range(2):
                if descendant_idx != feature_idx:
                    descendant = self._xtrain[:, descendant_idx]

                    # Calculate P(x_j | y, x_i = value)
                    mask = (feature_idx == value) & (self._ytrain == c)
                    total = np.sum(mask)

                    if total > 0:
                        prob_descendant_given_c_feature = np.sum(descendant[mask]) / total
                    else:
                        prob_descendant_given_c_feature = 0

                    self.cpts["descendants"][descendant][feature_idx][c][value] = prob_descendant_given_c_feature
