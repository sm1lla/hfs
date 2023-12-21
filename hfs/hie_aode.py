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

        self.cpts = []
        for sample in range(self._xtest.shape[0]):
            x_i = self._xtest[sample]
            class_prior = np.zeros((self.n_classes, self.n_features, 2))

            for feature_idx in range(len(x_i)):
                for c in range(self.n_classes):
                    for v in range(2):
                        class_prior[c][feature_idx][v] = (
                            np.sum(self._ytrain == c and x_i == v)
                            / self._ytrain.shape[0]
                        )

                ancestors = nx.ancestors(self._hierarchy, feature_idx)
                n_ancestors = len(ancestors)
                n_descendents = len(nx.descendants(self._hierarchy, feature_idx))

                ancestors_class_cpt = [
                    np.zeros((self.n_classes, self.n_features, 2))
                    for _ in range(n_ancestors)
                ]

                for a in range(n_ancestors):
                    for c in range(self.n_classes):
                        for v in range(2):
                            ancestors_class_cpt[a][c][feature_idx][v] = np.sum(
                                self._ytrain == c
                                and self._xtrain[sample, feature_idx] == v
                            ) / np.sum(self._ytrain == c)

                descendents_class_cpt = [
                    np.zeros((self.n_classes, self.n_features, 2))
                    for _ in range(n_descendents)
                ]

                # Todo calculate descendents

            self.cpts.append(
                {
                    "prior": class_prior,
                    "ancestors": ancestors_class_cpt,
                    "descendents": descendents_class_cpt,
                }
            )
