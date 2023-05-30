"HNB-select feature selection"

import numpy as np
from sklearn.naive_bayes import BernoulliNB
from filter import Filter


class HNBs(Filter):

    """
    Select non-redundant features following the algorithm proposed by Wan and Freitas 
    """

    def select_and_predict(self, predict = True, saveFeatures = False, estimator = BernoulliNB()):
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
            self._get_nonredundant_features_relevance(idx)
            if predict:
                predictions = np.append(predictions, self._predict(idx, estimator)[0])
            if saveFeatures:
                self._features[idx] = np.array(list(self._instance_status.values()))
                # self._features = np.vstack((np.array(list(self._instance_status.values())), self._features)) (but appending to np is very inefficient)
        return predictions
    