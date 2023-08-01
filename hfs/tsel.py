import numpy as np
from networkx.algorithms.dag import descendants
from sklearn.utils.validation import check_X_y

from hfs.feature_selection import EagerHierarchicalFeatureSelector
from hfs.helpers import get_paths
from hfs.metrics import lift


class TSELSelector(EagerHierarchicalFeatureSelector):
    """A tree-based feature selection method for hierarchical features proposed by Jeong and Myaeng"""

    def __init__(
        self, hierarchy: np.ndarray = None, use_original_implementation: bool = True
    ):
        super().__init__(hierarchy)
        self.use_original_implementation = use_original_implementation

    # TODO : check if columns parameter is really needed and think about how input should look like
    def fit(self, X, y, columns=None):
        """Fitting function that sets self.representatives_ to include the columns that are kept.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int, that should either be 1 or 0.
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)

        super().fit(X, y, columns)

        # Feature Selection Algorithm
        paths = get_paths(self._feature_tree)
        lift_values = lift(X, y)
        self._node_to_lift = {
            column_name: lift_values[index]
            for index, column_name in enumerate(self._columns)
        }
        self.representatives_ = self._find_representatives(paths)

        self.is_fitted_ = True
        return self

    def _find_representatives(self, paths: list[list[str]]):
        representatives = set()
        for path in paths:
            path.remove("ROOT")
            max_node = (
                self._select_from_path1(path)
                if self.use_original_implementation
                else self._select_from_path2(path)
            )
            representatives.add(max_node)
        return self._filter_representatives(representatives)

    def _select_from_path1(self, path: list[str]):
        # implementation used in paper by Jeong and Myaeng
        for index, node in enumerate(path):
            if index == len(path) - 1:
                return node
            elif self._node_to_lift[node] >= self._node_to_lift[path[index + 1]]:
                return node

    def _select_from_path2(self, path: list[str]):
        # if multiple nodes are maximum the node closest to the root is returned
        max_node = max(path, key=lambda x: self._node_to_lift[x])
        return max_node

    def _filter_representatives(self, representatives: list[str]):
        updated_representatives = []
        for node in representatives:
            selected_decendents = [
                descendent
                for descendent in descendants(self._feature_tree, node)
                if descendent in representatives
            ]
            if not selected_decendents:
                updated_representatives.append(node)
        return updated_representatives
