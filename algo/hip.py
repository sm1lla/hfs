import numpy as np
from sklearn.naive_bayes import BernoulliNB
from fixtures import getFixedDag, getFixedData
from filter import Filter



class HIP(Filter):

    """
    Select the k non-redundant features with the highest relevance following the algorithm proposed by Wan and Freitas 
    """
        
    def __init__(self, graph_data=None, k=0):

        super(HIP, self).__init__(graph_data)
        self.k = k

    def select_and_predict(self, predict = True, saveFeatures = False, estimator = BernoulliNB()):
        """
        Select features lazy for each test instance amd optionally predict target value of test instances.

        Parameters
        ----------
        predict :   {bool}
            true if predictions shall be obtained
        saveFeatures: {bool}
            true if features selected for each test instance shall be saved.
        estimator
                    Estimator to use for predictions
        

        Returns
        -------
        predictions for test input samples, if predict = false, returns empty array
        """
        predictions = np.array([])
        for idx in range(len(self._xtest)):
            self._get_nonredundant_features_mrt(idx)
            if predict:
                predictions = np.append(predictions, self._predict(idx, estimator)[0])
            if saveFeatures:
                self._features[idx] = np.array(list(self._instance_status.values()))
            for node in self._digraph:
                self._instance_status[node] = 1
        return predictions
    


