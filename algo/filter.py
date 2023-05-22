

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
import networkx as nx
from sklearn.naive_bayes import BernoulliNB
from fixtures import getFixedData, getFixedDag

from helpers import getRelevance

from abc import ABC


class Filter(BaseEstimator, ABC):
    """
    Abstract class used for all filter methods.

    Every method should implement the method select_and_predict.
    """

    def __init__(self, graph_data=None, estimator = BernoulliNB()): #todo G = None
        """
        Initialize a Filter with the required data.

        Parameters
        ----------
        graph_data 
                    {Numpy Array} of directed acyclic graph
        estimator
                    Estimator to use for predictions
        """
        self.graph_data = graph_data
        self.estimator = estimator

    def __get_relevance(self, node):
        """
        Gather relevance for a given node.

        Parameters
        ----------
        node
            Node for which the relevance should be obtained.
        """
        return getRelevance(self._xtrain, self._ytrain, node)
    
    def __get_ancestors(self, node):
        """
        Gather all ancestors for a given node.

        Parameters
        ----------
        node
            Node for which the ancestors should be obtained.
        """
        return nx.ancestors(self._digraph, node)
    
    def __get_descendants(self, node):
        """
        Gather all descendants for a given node.

        Parameters
        ----------
        node
            Node for which the descendants should be obtained.
        """
        return nx.descendants(self._digraph, node)
    
    def __create_digraph(self):
        """"
        Create digraph from numpy array.
        """
        self._digraph =  nx.from_numpy_array(self.graph_data, parallel_edges = False, create_using = nx.DiGraph)

    def _get_sorted_relevance(self):
        """
        Sort the nodes by descending relevance.
        """
        self._sorted_relevance = sorted(self._relevance, key=self._relevance.get)
    
    def fit(self, X, y=None):

        """
        Implementing the fit function for Sklearn Compatibility.

        Parameters
        ----------
        X : The training input samples.
        y: The target values, i.e., hierarchical class labels for classification.
    
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(
            X,
        )
        return self 
    
    def fit_selector(self, X_train, y_train, X_test, predict = True):
        """
        Fit Filter class. Due to laziness fitting of parameters as well as predictions are obtained per instance.

        Parameters
        ----------
        X_train : {numpy array} of shape (n_samples, n_features)
            The training input samples.
        X_test : {numpy array} of shape (n_samples, n_features)
            The test input samples. 
            converted into a sparse ``csc_matrix``.
        y_train : array-like of shape (n_samples, n_levels)
            The target values, i.e., hierarchical class labels for classification.
        """
        # Create DAG
        self.__create_digraph()
        self._xtrain = X_train
        self._ytrain = y_train
        self._xtest = X_test
        self._features = np.zeros(shape=X_test.shape)

        # Get relevance, ancestors and descendants of each node
        self._relevance = {}
        self._descendants = {}
        self._ancestors = {}
        for node in self._digraph:
            self._relevance[node] =  self.__get_relevance(node)
            self._ancestors[node] =  self.__get_ancestors(node)
            self._descendants[node] =  self.__get_descendants(node)
        self._get_sorted_relevance()

        self._instance_status = {}
        for node in self._digraph:
            self._instance_status[node] = 1
    
    
    def select_and_predict(self, predict = True, saveFeatures = False):
       pass

    def _get_nonredundant_features(self, idx):
        """
        Get nonredundant features. Basic functionality of the algorithm poropes by Wan & Freitas.

        Parameters
        ----------
        idx
            Index of test instance for which the features shall be selected.
        """
        for node in self._digraph:
            if self._xtest[idx][node] == 1:
                for anc in self._ancestors[node]:
                    if self._relevance[anc] <= self._relevance[node]:
                        self._instance_status[anc] = 0
            else:
                for desc in self._descendants[node]:
                    if self._relevance[desc] <= self._relevance[node]:
                        self._instance_status[desc] = 0
    
    def _get_top_k(self):
        """
        Get k highest-ranked features by relevance.

        """
        counter = 0
        for node in self._sorted_relevance:
            if (counter < self.k or not self.k) and self._instance_status[node]:
                counter+=1
            else:
                self._instance_status[node] = 0
 

    def _predict(self, idx):
        """
        Predicts for .

        Parameters
        ----------
        idx
            Index of test instance which shall be predicted.

        Returns
        -------
        prediction : bool
            prediction of test instance's target value.
        """
        features = [nodes for nodes, status in self._instance_status.items() if status]
        clf = self.estimator
        clf.fit(self._xtrain[:,features], self._ytrain)
        return clf.predict(self._xtest[idx][features].reshape(1, -1))
    
    def score(self, ytest, predictions):
        print(accuracy_score(y_true = ytest, y_pred=predictions))

    def get_features(self):
        return self._features


#todo:

#pipeline + wofür? macht eigentlich keinen Sinn, dafür wäre tatsächlich fit und transform notwendig.
#test in fit ob DAten korrekt für DAG
#kindklassen, +
#optionen: predict, feature-ausgabe + 
#test!
#dokumentation und klassendokumentation +
