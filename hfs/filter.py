

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
import networkx as nx
from sklearn.naive_bayes import BernoulliNB

from helpers import checkData, getRelevance

from abc import ABC


class Filter(BaseEstimator, ABC):
    """
    Abstract class used for all filter methods.

    Every method should implement the method select_and_predict.
    """

    def __init__(self, graph_data=None): #todo G = None
        """
        Initialize a Filter with the required data.

        Parameters
        ----------
        graph_data 
                    {Numpy Array} of directed acyclic graph
        """
        self.graph_data = graph_data

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
    
    def _create_digraph(self):
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
    
    def fit_selector(self, X_train, y_train, X_test):
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
        self._create_digraph()
        self._xtrain = X_train
        self._ytrain = y_train
        self._xtest = X_test
        self._features = np.zeros(shape=X_test.shape)

        # Validate data
        checkData(self._digraph, self._xtrain, self._ytrain)
        #checkData(self._digraph, self._xtest, self._ytest) ???

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
    
    
    def select_and_predict(self, predict = True, saveFeatures = False, estimator = BernoulliNB()):
       pass

    def _get_nonredundant_features(self, idx):
        """
        Get nonredundant features without relevance score. 
        Basic functionality of the algorithm HIP proposed by Wan & Freitas.

        Parameters
        ----------
        idx
            Index of test instance for which the features shall be selected.
        """
        for node in self._digraph:
            if self._xtest[idx][node] == 1:
                for anc in self._ancestors[node]:
                    self._instance_status[anc] = 0
            else:
                for desc in self._descendants[node]:
                    self._instance_status[desc] = 0
              
    def _get_nonredundant_features_relevance(self, idx):
        """
        Get nonredundant features based on relevance score. 
        Basic functionality of the HNB algorithm proposed by Wan & Freitas.

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
                        

    def get_nonredundant_features_mrt(self, idx):
        """
        Get nonredundant features based on the MRT considering all pathes. 
        Basic functionality of the HIP algorithm proposed by Wan & Freitas.

        Parameters
        ----------
        idx
            Index of test instance for which the features shall be selected.
        """
        
        top_sort = nx.topological_sort(self._digraph)
        mrt = {}

        for node in top_sort:

            # for each node save highest seen node in path to node
            mrt[node] = node

                # correctness: as each predecessor lies on the same path as the current node
                # one can be removed.
                #
                # not several nodes per path: 
                # contradiction - nodes a and b (a<b) selected and are on the same path.
                # each node a+1..b-1 between would save a as most relevant node:
                # 0: mrt[a] = a
                # 1: mrt[a+1] = a+1, if rel[mrt[pred]=a] > rel[mrt[a+1]=a+1]: mrt[a+1]=a 
                # and status[a+1] = remove
                # i: mrt[a+i] = a+i, if rel[mrt[pred]=a] > rel[mrt[a+i]=a+i]: mrt[a+i]=a 
                # and status[a+i] = remove
                # When b is processed, mrt[b-1] = a, so either rel[a]>rel[b] -> mrt[b]=b is removed
                # and set to a OR mrt[b-1]=a is removed and mrt[b] stays b.
                #
                # at least one node per path: As there is a maximum relevant node in each path
                # this node will stay mrt[node] = node and not be exchanged through mrt[pred].
                # Then the condition is never met on this path
                # so self._instance_status[mrt[node]] = 0 never executed.
                #
            if self._xtest[idx][node]:

                for pred in self._digraph.predecessors(node):
                    # get most relevant node seen on path until current node
                    if self._relevance[mrt[pred]] > self._relevance[mrt[node]]:
                        # each node not selected will removed
                        self._instance_status[mrt[node]] = 0
                        mrt[node] = mrt[pred]
                    else:
                        self._instance_status[mrt[pred]] = 0
       
            if not self._xtest[idx][node]:
                for suc in self._digraph.successors(node):
                    # get most relevant node seen on path until current node
                    if self._relevance[mrt[suc]] > self._relevance[mrt[node]]:
                        # each node not selected will removed
                        self._instance_status[mrt[node]] = 0
                        mrt[node] = mrt[suc]
                    else:
                        self._instance_status[mrt[suc]] = 0


    
    def _get_top_k(self):
        """
        Get k highest-ranked features by relevance.

        """
        counter = 0
        for node in reversed(self._sorted_relevance):
            if (counter < self.k or not self.k) and self._instance_status[node]:
                counter+=1
            else:
                self._instance_status[node] = 0
 

    def _predict(self, idx, estimator):
        """
        Predicts for .

        Parameters
        ----------
        idx
            Index of test instance which shall be predicted
        estimator
                    Estimator to use for predictions.

        Returns
        -------
        prediction : bool
            prediction of test instance's target value.
        """
        features = [nodes for nodes, status in self._instance_status.items() if status]
        clf = estimator
        clf.fit(self._xtrain[:,features], self._ytrain)
        return clf.predict(self._xtest[idx][features].reshape(1, -1))
    
    def get_score(self, ytest, predictions):
        return accuracy_score(y_true = ytest, y_pred=predictions)

    def get_features(self):
        return self._features
