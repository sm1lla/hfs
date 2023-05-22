import numpy as np
from sklearn.naive_bayes import BernoulliNB
from fixtures import getFixedDag, getFixedData
from filter import Filter



class HNB(Filter):

    """
    Select the k non-redundant features with the highest relevance following the algorithm proposed by Wan and Freitas 
    """
        
    def __init__(self, graph_data=None, k=0, estimator = BernoulliNB()):

        super(HNB, self).__init__(graph_data, estimator)
        self.k = k

    def select_and_predict(self, predict = True, saveFeatures = False):
        """
        Select features lazy for each test instance amd optionally predict target value of test instances.

        Parameters
        ----------
        predict :   {bool}
            true if predictions shall be obtained
        saveFeatures: {bool}
            true if features selected for each test instance shall be saved.
        

        Returns
        -------
        predictions for test input samples, if predict = false, returns empty array
        """
        predictions = np.array([])
        for idx in range(len(self._xtest)):
            self._get_nonredundant_features(idx)
            self._get_top_k()
            if predict:
                predictions = np.append(predictions, self._predict(idx)[0])
            if saveFeatures:
                self._features[idx] = np.array(list(self._instance_status.values()))
                # self._features = np.vstack((np.array(list(self._instance_status.values())), self._features)) (but appending to np is very inefficient)
        return predictions

class HNBs(Filter):

    """
    Select non-redundant features following the algorithm proposed by Wan and Freitas 
    """

    def select_and_predict(self, predict = True, saveFeatures = False):
        """
        Select features lazy for each test instance amd optionally predict target value of test instances.

        Parameters
        ----------
        predict :   {bool}
            true if predictions shall be obtained
        saveFeatures: {bool}
            true if features selected for each test instance shall be saved.
        

        Returns
        -------
        predictions for test input samples, if predict = false, returns empty array
        """
        predictions = np.array([])
        for idx in range(len(self._xtest)):
            self._get_nonredundant_features(idx)
            if predict:
                predictions = np.append(predictions, self._predict(idx)[0])
            if saveFeatures:
                self._features[idx] = np.array(list(self._instance_status.values()))
                # self._features = np.vstack((np.array(list(self._instance_status.values())), self._features)) (but appending to np is very inefficient)
        return predictions
    
class RNB(Filter):

    """
    Select the k features with the highest relevance

    """

    def __init__(self, graph_data=None, k=0, estimator = BernoulliNB()):

        super(RNB, self).__init__(graph_data, estimator)
        self.k = k

    def select_and_predict(self, predict = True, saveFeatures = False):
        """
        Select features lazy for each test instance amd optionally predict target value of test instances.

        Parameters
        ----------
        predict :   {bool}
            true if predictions shall be obtained
        saveFeatures: {bool}
            true if features selected for each test instance shall be saved.
        

        Returns
        -------
        predictions for test input samples, if predict = false, returns empty array
        """
        predictions = np.array([])
        for idx in range(len(self._xtest)):
            self._get_top_k() #change as equal for each test instance
            if predict:
                predictions = np.append(predictions, self._predict(idx)[0])
            if saveFeatures:
                self._features[idx] = np.array(list(self._instance_status.values()))
                # self._features = np.vstack((np.array(list(self._instance_status.values())), self._features)) (but appending to np is very inefficient)
        return predictions
    

    


    
x = HNBs(getFixedDag())
x_t, y_t = getFixedData(50)
x_test, y_test = getFixedData(5)    
x.fit_selector(x_t, y_t, x_test)
pred = x.select_and_predict(True, True)
print(x.score(y_test, pred))
fe = x.get_features()
#print(fe)