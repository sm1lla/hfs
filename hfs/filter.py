from abc import ABC

import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB

from .base import HierarchicalEstimator
from .helpers import checkData, conditional_mutual_information, getRelevance


class Filter(HierarchicalEstimator, ABC):
    """
    Abstract class used for all filter methods.

    Every method should implement the method select_and_predict.
    """

    def __init__(self, hierarchy: np.ndarray = None):  # todo G = None
        """
        Initialize a Filter with the required data.

        Parameters
        ----------
        hierarchy
                    {Numpy Array} of directed acyclic graph
        """
        self.hierarchy = hierarchy

    def _get_relevance(self, node):
        """
        Gather relevance for a given node.

        Parameters
        ----------
        node
            Node for which the relevance should be obtained.
        """
        return getRelevance(self._xtrain, self._ytrain, node)

    def _get_ancestors(self, node):
        """
        Gather all ancestors for a given node.

        Parameters
        ----------
        node
            Node for which the ancestors should be obtained.
        """
        return nx.ancestors(self._feature_tree, node)

    def _get_descendants(self, node):
        """
        Gather all descendants for a given node.

        Parameters
        ----------
        node
            Node for which the descendants should be obtained.
        """
        return nx.descendants(self._feature_tree, node)

    def _create_feature_tree(self):
        """ "
        Create digraph from numpy array.
        """
        self._feature_tree = nx.from_numpy_array(
            self.hierarchy, parallel_edges=False, create_using=nx.DiGraph
        )

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
        self.n_features_ = X.shape[1]
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
        self._set_feature_tree()
        self._feature_tree.remove_node("ROOT")
        self._xtrain = X_train
        self._ytrain = y_train
        self._xtest = X_test
        self.n_features_ = X_train.shape[1]
        self._features = np.zeros(shape=X_test.shape)

        # Validate data
        checkData(self._feature_tree, self._xtrain, self._ytrain)
        # checkData(self._feature_tree , self._xtest, self._ytest) ???

        # Get relevance, ancestors and descendants of each node
        self._relevance = {}
        self._descendants = {}
        self._ancestors = {}
        for node in self._feature_tree:
            self._relevance[node] = self._get_relevance(node)
            self._ancestors[node] = self._get_ancestors(node)
            self._descendants[node] = self._get_descendants(node)
        self._get_sorted_relevance()

        self._instance_status = {}
        for node in self._feature_tree:
            self._instance_status[node] = 1

    def select_and_predict(
        self, predict=True, saveFeatures=False, estimator=BernoulliNB()
    ):
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
        for node in self._feature_tree:
            if self._xtest[idx][node] == 1:
                for anc in self._feature_tree.predecessors(node): #TODO: save it first to make it more efficient?
                    self._instance_status[anc] = 0
            else:
                for desc in self._feature_tree.successors(node):
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
        for node in self._feature_tree:
            if node == "ROOT":
                continue
            if self._xtest[idx][node] == 1:
                for anc in self._ancestors[node]:
                    if self._relevance[anc] <= self._relevance[node]:
                        self._instance_status[anc] = 0
            else:
                for desc in self._descendants[node]:
                    if self._relevance[desc] <= self._relevance[node]:
                        self._instance_status[desc] = 0

    def _get_nonredundant_features_mrt(self, idx):
        """
        Get nonredundant features based on the MRT considering all pathes.
        Basic functionality of the HIP algorithm proposed by Wan & Freitas.

        Parameters
        ----------
        idx
            Index of test instance for which the features shall be selected.
        """

        top_sort = list(nx.topological_sort(self._feature_tree))
        reverse_top_sort = reversed(top_sort)
        mrt = {}

        for node in top_sort:
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
            mrt[node] = []
            more_rel_nodes = [node]
            if self._xtest[idx][node]:
                # preds are 1 because of 0-1-propagation
                for pred in self._feature_tree.predecessors(node):
                    # get most relevant nodes seen on the paths until current node
                    for _mrt in mrt[pred]:
                        # if their is a node on the path more important then current node
                        if self._relevance[_mrt] > self._relevance[node]:
                            self._instance_status[node] = 0
                            # save this node for next iterations (steps on path)
                            more_rel_nodes.append(_mrt)
                        else:
                            # save current node as most important.
                            # there can be several paths, in this case, several nodes are saved
                            self._instance_status[_mrt] = 0
                            more_rel_nodes.append(node)
            mrt[node] = more_rel_nodes

        for node in reverse_top_sort:
            mrt[node] = []
            more_rel_nodes = [node]
            if not self._xtest[idx][node]:
                mrt[node] = node
                for suc in self._feature_tree.successors(node):
                    # get most relevant nodes seen on paths until current node
                    for _mrt in mrt[suc]:
                        if self._relevance[_mrt] > self._relevance[node]:
                        # each node not selected will removed
                            self._instance_status[node] = 0
                            more_rel_nodes.append(_mrt)
                        else:
                            self._instance_status[_mrt] = 0
                            more_rel_nodes.append(node)
            mrt[node] = more_rel_nodes

    def _get_top_k(self):
        """
        Get k highest-ranked features by relevance.

        """
        counter = 0
        for node in reversed(self._sorted_relevance):
            if node == "ROOT":
                continue
            if (counter < self.k or not self.k) and self._instance_status[node]:
                counter += 1
            else:
                self._instance_status[node] = 0

    def _build_mst(self):
        edges = self._feature_tree.edges
        self._edge_status = np.zeros((self.n_features_,self.n_features_))
        self._cmi = np.zeros((self.n_features_,self.n_features_))
        self._sorted_edges = []
        for node1 in self._feature_tree.nodes:
            for node2 in self._feature_tree.nodes:
                if node1 == node2:
                    continue
                self._cmi[node1][node2] = conditional_mutual_information(self._xtrain[:,node1], self._xtrain[:,node2], self._ytrain)
                self._edge_status[node1][node2] = 1
        sorted_indices = np.argsort(self._cmi, axis = None)
        for index in sorted_indices:
            coordinates = divmod(index, self.n_features_)

            if coordinates[0] < coordinates[1]:
                self._sorted_edges.append(coordinates)
            

        

    def _get_nonredundant_features_from_mst(self, idx):
        """
        Get nonredundant features from MST.
        Basic functionality of the algorithm TAN proposed by Wan & Freitas.

        Parameters
        ----------
        idx
            Index of test instance for which the features shall be selected.
        """
        UDAG = nx.Graph()

        for node1 in self._feature_tree:
            self._instance_status[node1] = 0
            for node2 in self._feature_tree:
                    self._edge_status[node1][node2] = 1

        representants = [i for i in range(self.n_features_)]
        members = {}
        for i in range(self.n_features_):
            members[i]=[i]

        # get paths 
        reachable_nodes = {}
        for node in self._feature_tree:
            reachable_nodes[node] = []
            for des in nx.descendants(self._feature_tree, node):
                reachable_nodes[node].append(des)
        # select edges
        for edge in self._sorted_edges:
            if (self._edge_status[edge[0]][edge[1]]
                 # check redundancy: same path and same value
                and (self._xtest[idx][edge[0]] != self._xtest[idx][edge[1]] or 
                     (edge[0] not in reachable_nodes[edge[1]] and edge[1] not in reachable_nodes[edge[1]]))
                # check if circle in UDAG using the property, that edge (a,b) infers circle iff a und b
                # are members of the same component
                and representants[edge[0]] != representants[edge[1]]):
                
                UDAG.add_edge(edge[0], edge[1])
                self._edge_status[edge[0]][edge[1]] = 0

                # merge: change the representatives of the smaller component
                if len(members[representants[edge[0]]]) <= len(members[representants[edge[1]]]):
                    for m in members[edge[0]]:
                        representants[m] = representants[edge[1]]
                        members[representants[edge[1]]].append(m)
                else:
                    for m in members[edge[1]]:
                        representants[m] = representants[edge[0]]
                        members[representants[edge[0]]].append(m)

                # remove all edges with redundant ancestors or descendants of e0 and e1
                for selected_node in [edge[0], edge[1]]:
                    for neighbor_node in nx.ancestors(self._feature_tree, selected_node):
                        if self._xtest[idx][selected_node] == self._xtest[idx][neighbor_node]:
                            # alternative: collect all and then delete in sorted_edges
                            self._edge_status[:,neighbor_node] = 0
                            self._edge_status[neighbor_node][:] = 0
                    for neighbor_node in nx.descendants(self._feature_tree, selected_node):
                        if self._xtest[idx][selected_node] == self._xtest[idx][neighbor_node]:
                            self._edge_status[:,neighbor_node] = 0
                            self._edge_status[neighbor_node][:] = 0

                self._instance_status[edge[0]] = 1
                self._instance_status[edge[1]] = 1
                
            

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
        clf.fit(self._xtrain[:, features], self._ytrain)
        return clf.predict(self._xtest[idx][features].reshape(1, -1))

    def get_score(self, ytest, predictions):
        return accuracy_score(y_true=ytest, y_pred=predictions)

    def get_features(self):
        return self._features
