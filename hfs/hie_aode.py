"HNB-select feature selection"

import networkx as nx
import numpy as np
from sklearn.naive_bayes import BernoulliNB

from .lazyHierarchicalFeatureSelector import LazyHierarchicalFeatureSelector


class HieAODE(LazyHierarchicalFeatureSelector):
    """
    Select non-redundant features following the algorithm proposed by Wan and Freitas.
    """

    def __init__(self, hierarchy=None):
        """Initializes a HNBs-Selector.

        Parameters
        ----------
        hierarchy : np.ndarray
            The hierarchy graph as an adjacency matrix.
        """
        super(HieAODE, self).__init__(hierarchy)

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
        self.n_classes
        self.n_features

        cpts = []
        for i, x_i in enumerate(self._xtest):
            n_ancestors = nx.ancestors(self._hierarchy, i)
            n_descendents = nx.descendants(self._hierarchy, i)
            class_prior = np.zeros(self.n_classes, self.n_features, 2)

            ancestors_class_cpt = [
                np.zeros(self.n_classes, self.n_features, 2)
                for _ in range(n_ancestors)
            ]

            descendents_class_cpt = [
                np.zeros(self.n_classes, self.n_features, 2)
                for _ in range(n_descendents)
            ]

            for feature in x_i:


            cpts.append(
                {
                    "prior": class_prior,
                    "ancestors": ancestors_class_cpt,
                    "descendents": descendents_class_cpt,
                }
            )
