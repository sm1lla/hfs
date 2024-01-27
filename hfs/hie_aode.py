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
        ancestors_class_cpt = (self.n_ancestors, self.n_classes, self.n_features, n_values)

        P (x_j|y, x_i)
        feature_descendants_class_cpt = (self.n_features, self.n_descendants, self.n_classes, n_values)
        """
        super(HieAODE, self).fit_selector(X_train, y_train, X_test, columns)
        self.cpts = dict(
            # prior = P(y, x_i)
            prior=np.full((self.n_features, # x_i
                           self.n_classes,  # y
                           2  # value
                           ), -1, dtype=float),
            #(x_j (descendent), x_i (current feature), class, value)  )
            descendants=np.full(
                (self.n_features, self.n_features, self.n_classes, 2, 2), -1,dtype=float
            ),  # P(x_j|y, x_i)
            ancestors=np.full((self.n_features, self.n_classes, 2), -1, dtype=float),  # P(x_k|y)
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
        n_samples = self._xtest.shape[0]
        sample_sum = np.zeros((n_samples, self.n_classes))
        for sample_idx in range(n_samples):
            sample = self._xtest[sample_idx]

            descendant_product = np.ones(self.n_classes)
            ancestor_product = np.ones(self.n_classes)
            for feature_idx in range(len(sample)):
                self.calculate_class_prior(feature_idx=feature_idx)

                ancestors = list(nx.ancestors(self._hierarchy, feature_idx))
                for ancestor_idx in ancestors:
                    self.calculate_prob_given_ascendant_class(ancestor=ancestor_idx)

                descendants = list(nx.descendants(self._hierarchy, feature_idx))
                # P (x_j=sample[descendant_idx]|y, x_i=sample[feature_idx])
                for descendant_idx in descendants:
                    self.calculate_prob_descendant_given_class_feature(
                        descendant_idx=descendant_idx, feature_idx=feature_idx
                    )

                if len(ancestors) <= 0:
                    ancestor_product = np.zeros((self.n_classes))
                else:
                    ancestor_product = np.prod(
                        self.cpts["ancestors"][ancestors, :, sample[ancestors]], axis=0
                    )
                if len(descendants) <= 0:
                    descendant_product = np.zeros((self.n_classes))
                else:
                    descendant_product = np.prod(
                        self.cpts["descendants"][descendants, feature_idx, :, sample[feature_idx], sample[descendants]], axis=0
                    )

                feature_prior = np.prod(
                    self.cpts["prior"][feature_idx, :, sample[feature_idx]
                    ])

                feature_product = np.multiply(ancestor_product, descendant_product)
                feature_product = np.multiply(feature_product, feature_prior)
                
                sample_sum[sample_idx] = np.add(sample_sum[sample_idx], feature_product)

        y = np.argmax(sample_sum, axis=1)
        return y if predict else np.array([])


    def calculate_class_prior(self, feature_idx):
        n_samples = self._ytrain.shape[0]
        for c in range(self.n_classes):
            for value in range(2):
                if self.cpts["prior"][feature_idx][c][value] == -1:
                    value_sum = np.sum((self._ytrain == c) & (self._xtrain[:,feature_idx] == value))
                    self.cpts["prior"][feature_idx][c][value] = value_sum / n_samples

    def calculate_prob_given_ascendant_class(self, ancestor):
        # Calculate P(x_k | y) where x_k=ascendant and y = c
        for c in range(self.n_classes):
            p_class = np.sum(self._ytrain == c)
            for value in range(2):
                p_class_ascendant = np.sum(
                    (self._ytrain == c) & (self._xtrain[:, ancestor] == value)
                )
                self.cpts["ancestors"][ancestor][c][value] = (
                    p_class_ascendant / p_class
                )

    def calculate_prob_descendant_given_class_feature(
        self, descendant_idx, feature_idx
    ):
        for c in range(self.n_classes):
            for feature_value in range(2):
                # Calculate P(y, x_i = value)
                mask = (self._xtrain[:, feature_idx] == feature_value) & (self._ytrain == c)
                total = np.sum(mask)
                for descendant_value in range(2):
                    if descendant_idx != feature_idx:
                        if total > 0:
                            descendant = self._xtrain[:, descendant_idx]
                            d = np.sum(descendant[mask] == descendant_value)
                            prob_descendant_given_c_feature = d / total
                        else:
                            prob_descendant_given_c_feature = 0

                        self.cpts["descendants"][descendant_idx][feature_idx][c][descendant_value][
                            feature_value
                        ] = prob_descendant_given_c_feature
