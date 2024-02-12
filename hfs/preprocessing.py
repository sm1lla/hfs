"""
Sklearn compatible estimators for preprocessing hierarchical data.
"""
import networkx as nx
import numpy as np
from networkx.algorithms.dag import ancestors
from sklearn.utils.validation import check_array, check_is_fitted

from hfs.helpers import get_irrelevant_leaves
from hfs.selectors import HierarchicalEstimator


class HierarchicalPreprocessor(HierarchicalEstimator):
    """Estimator for preprocessing hierarchical data for feature selection.

    The hierarchical feature selectors expect the input data and the
    hierarchy graph to conform to certain pre-conditions.
    This preprocessor prepares the data and graph for the feature
    selection.
    """

    def __init__(self, hierarchy: np.ndarray = None):
        """Initializes a HierarchicalPreprocessor.

        Parameters
        ----------
        hierarchy : np.ndarray
                    The hierarchy graph as an adjacency matrix."""
        self.hierarchy = hierarchy

    def fit(self, X, y=None, columns=None):
        """Sets the parameters for data transformation and prepares hierarchy.

        Following conditions need to be fulfilled for the feature
        selection algorithms:
            - every node in the hierarchy graph should be able to be mapped to
              one column in the dataset and every column in the dataset should
              have a corresponding node in the hierarchy.
            - for binary data, if a feature has the value 1, all of its descendents
              in the hierarchy should also have the value 1.

        To achieve these conditions missing columns are added to
        the hierarchy and unnecessary nodes are removed. The self._columns
        parameter is adjusted so that it can be used to add additional columns
        to the dataset in the transform method. After fitting the dataset can
        be transformed with the transform method and the updated hierarchy
        and columns mapping can be retrieved with get_hierarchy and
        get_columns.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        columns: list or None, length n_features
            The mapping from the hierarchy graph's nodes to the columns in X.
            A list of ints. If this parameter is None the columns in X and
            the corresponding nodes in the hierarchy are expected to be in the
            same order.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=True)
        super().fit(X, y, columns)
        if columns is None:
            self._columns = [-1] * self.n_features_in_

        self._extend_dag()
        self._shrink_dag()
        self._find_missing_columns()
        self._adjust_node_names()
        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Transforms dataset to fulfill conditions for feature selection.

        After transformation, if a feature is 1, all of its descendents are 1.
        Missing columns are added to the dataset.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_ : array of shape (n_samples, n_selected_features)
            The transformed dataset.
        """
        # Check is fit had been called
        check_is_fitted(self, "is_fitted_")

        # Input validation
        X = check_array(X, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Shape of input is different from what was seen" "in `fit`"
            )

        X_ = self._add_columns(X)
        X_ = self._propagate_ones(X_)
        return X_

    def get_hierarchy(self):
        """Get the transformed hierarchy graph.

        Raises
        ----------
        RuntimeError
            If the method is called before fit has been called.
            In this case the hierarchy graph has not been updated yet.
        """
        if self.is_fitted_:
            output_hierarchy = self._hierarchy
            output_hierarchy.remove_node("ROOT")
            return nx.to_numpy_array(self._hierarchy)
        else:
            raise RuntimeError("Instance has not been fitted.")

    def _extend_dag(self):
        """Adds missing nodes to the hierarchy graph.

        For columns that no have a corresponding node in the hierarchy a
        node is added right under the "ROOT" node.
        """
        max = len(self._hierarchy.nodes) - 1
        for x in range(len(self._columns)):
            if self._columns[x] == -1:
                if x in self._hierarchy.nodes:
                    self._hierarchy.add_edge("ROOT", max)
                    self._columns[x] = max
                    max += 1
                else:
                    self._hierarchy.add_edge("ROOT", x)
                    self._columns[x] = x

    def _shrink_dag(self):
        """Unnecessary nodes are removed from the hierarchy graph.

        Nodes are considered unnecessary if they do not have a corresponding
        column in the input dataframe and don't have any children. These
        features would always be 0 in the dataset and, therefore, do not
        contain any necessary information.
        """
        leaves = get_irrelevant_leaves(
            x_identifier=self._columns, digraph=self._hierarchy
        )
        while leaves:
            for x in leaves:
                self._hierarchy.remove_node(x)
            leaves = get_irrelevant_leaves(
                x_identifier=self._columns, digraph=self._hierarchy
            )

    def _find_missing_columns(self):
        """Finds nodes for which a column needs to be added to the dataset.

        These node names are added to self._columns and the corresponding
        columns will be added in the transform method.
        """
        missing_nodes = [
            node
            for node in self._hierarchy.nodes
            if node not in self._columns and node != "ROOT"
        ]
        self._columns.extend(missing_nodes)

    def _add_columns(self, X):
        """Adds missing columns to the dataset.

        Missing columns are added and all values are set to 0.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_ : array of shape [n_samples, n_new_features]
            The dataset with the added columns.
        """
        X_ = X
        num_rows, num_columns = X.shape
        if num_columns < len(self._columns):
            missing_indices = list(range(num_columns, len(self._columns)))
            for _ in missing_indices:
                X_ = np.concatenate([X_, np.zeros((num_rows, 1), dtype=int)], axis=1)
        return X_

    def _propagate_ones(self, X):
        """Update the dataset to fulfill the 0-1-propagation rule..

        If a feature in the dataset in 1 all its descendents in the
        sample are set to 1.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X : array of shape [n_samples, n_new_features]
            The dataset with updated feature values.
        """
        nodes = list(self._hierarchy.nodes)
        nodes.remove("ROOT")

        for node in nodes:
            column_index = self._column_index(node)
            ancestor_nodes = ancestors(self._hierarchy, node)
            ancestor_nodes.remove("ROOT")
            for row_index, entry in enumerate(X[:, column_index]):
                if entry == 1.0:
                    for ancestor in ancestor_nodes:
                        index = self._column_index(ancestor)
                        X[row_index, index] = 1.0
        return X

    def _adjust_node_names(self):
        """Adjust node names in hierarchy and _columns.

        When nodes are removed from the hierarchy graph the mapping in
        self._columns is not correct anymore after the hierarchy graph
        is transformed to a numpy.ndarray and back again. However, this
        transformation needs to be performed to ouput the hierarchy.
        Therefore the node names need to be adjusted.
        """
        nodes = list(self._hierarchy.nodes())
        nodes.remove("ROOT")
        self._columns = [nodes.index(node_name) for node_name in self._columns]
        mapping = {node_name: nodes.index(node_name) for node_name in nodes}
        self._hierarchy = nx.relabel_nodes(self._hierarchy, mapping)
