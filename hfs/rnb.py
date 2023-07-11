"RNB feature selection"

import numpy as np
from sklearn.naive_bayes import BernoulliNB

from .filter import Filter


class RNB(Filter):

    """
    Select the k features with the highest relevance

    """

    def __init__(self, hierarchy=None, k=0):
        super(RNB, self).__init__(hierarchy)
        self.k = k

    def select_and_predict(
        self, predict=True, saveFeatures=False, estimator=BernoulliNB()
    ):
        """
        Select features lazy for each test instance amd optionally predict target value of test instances.

        Parameters
        ----------
        predict :   {bool}
            true if predictions shall be obtained
        saveFeatures: {bool}
            true if features selected for each test instance shall be saved
        estimator
                    Estimator to use for predictions.

        Returns
        -------
        predictions for test input samples, if predict = false, returns empty array
        """
        predictions = np.array([])
        for idx in range(len(self._xtest)):
            self._get_top_k()  # change as equal for each test instance
            if predict:
                predictions = np.append(predictions, self._predict(idx, estimator)[0])
            if saveFeatures:
                self._features[idx] = np.array(list(self._instance_status.values()))
            self._feature_length[idx] = len([nodes for nodes, status in self._instance_status.items() if status])
        return predictions
