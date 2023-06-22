import math

import numpy as np
from networkx.algorithms.dag import descendants
from scipy import sparse
from sklearn.utils.validation import check_X_y

from hfs.feature_selection import HierarchicalFeatureSelector
from hfs.helpers import cosine_similarity, get_leaves, normalize_score


class HillClimbingSelector(HierarchicalFeatureSelector):
    def __init__(
        self,
        hierarchy: np.ndarray = None,
        alpha: float = 0.99,
        dataset_type: str = "binary",
    ):
        super().__init__(hierarchy)
        self.alpha = alpha
        self.dataset_type = dataset_type

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
        # Input validation
        X, y = check_X_y(X, y, accept_sparse=True)

        super().fit(X, y, columns)
        if sparse.issparse(X):
            X = X.tocsr()

        # Feature Selection Algorithm
        self.y_ = y
        self._num_rows = X.shape[0]
        self.representatives_ = self._hill_climb(X)

        return self

    def _hill_climb(self, X):
        raise NotImplementedError

    def _calculate_scores(self, X):
        score_matrix = np.zeros((self._num_rows, self.n_features_))
        for row in range(self._num_rows):
            for column_index in range(self.n_features_):
                children = []
                if self._columns[column_index in self._feature_tree.nodes]:
                    children = list(
                        descendants(self._feature_tree, self._columns[column_index])
                    )
                scores_children = [X[row, child] for child in children]
                score = sum(scores_children, start=X[row, column_index])

                if self.dataset_type == "numerical":
                    normalize_score(score, max(X[row, :]))

                score_matrix[row, column_index] = score
        return score_matrix

    def _calculate_distance(
        self,
        sample_i: int,
        sample_j: int,
        feature_set: list[int],
    ):
        raise NotImplementedError

    def _calculate_distances(self, feature_set: list[int]):
        distances = np.zeros((self._num_rows, self._num_rows), dtype=float)
        for row in range(self._num_rows):
            for column in range(self._num_rows):
                if distances[row, column] == 0:
                    distances[row, column] = self._calculate_distance(
                        row, column, feature_set
                    )
                    distances[column, row] = distances[row, column]
        return distances

    def _fitness_function(self, distances: np.ndarray) -> float:
        raise NotImplementedError


class TopDownSelector(HillClimbingSelector):
    """Hill climbing feature selection method for hierarchical features proposed by Wang et al.2002"""

    def __init__(
        self,
        hierarchy: np.ndarray = None,
        alpha: float = 0.99,
        dataset_type: str = "binary",
    ):
        super().__init__(hierarchy, alpha=alpha, dataset_type=dataset_type)

    def _hill_climb(self, X) -> list[int]:
        self._score_matrix = self._calculate_scores(X)
        optimal_feature_set = set(self._feature_tree.successors("ROOT"))
        fitness = 0
        best_fitness = 0
        best_feature_set = None

        while True:
            for node in optimal_feature_set:
                children = list(self._feature_tree.successors(node))
                if children:
                    temporary_feature_set = optimal_feature_set.copy()
                    temporary_feature_set.remove(node)
                    temporary_feature_set.update(children)
                    distances = self._calculate_distances(temporary_feature_set)
                    temporary_fitness = self._fitness_function(distances)
                    if (temporary_fitness) > best_fitness:
                        best_fitness = temporary_fitness
                        best_feature_set = temporary_feature_set

            if best_fitness > fitness:
                optimal_feature_set = best_feature_set
                fitness = best_fitness
            else:
                break
        return list(optimal_feature_set)

    def _calculate_distance(
        self,
        sample_i: int,
        sample_j: int,
        feature_set: list[int],
    ):
        distance = 0
        for column in feature_set:
            column_index = self._column_index(column)
            difference = (
                self._score_matrix[sample_i, column_index]
                - self._score_matrix[sample_j, column_index]
            )
            distance += math.pow(difference, 2)
        return math.sqrt(distance)

    def _fitness_function(self, distances: np.ndarray) -> float:
        result = 0
        row_indices = range(self._num_rows)
        for row_index in row_indices:
            same_class = [
                sample
                for sample in row_indices
                if self.y_[sample] == self.y_[row_index]
            ]
            other_class = [sample for sample in row_indices if sample not in same_class]

            nominator = sum([distances[sample, row_index] for sample in other_class])
            denominator = 1 + self.alpha * sum(
                [distances[sample, row_index] for sample in same_class]
            )
            result += nominator / denominator

        return result


class BottomUpSelector(HillClimbingSelector):
    """Hill climbing feature selection method for hierarchical features proposed by Wang et al.2003"""

    def __init__(
        self,
        hierarchy: np.ndarray = None,
        alpha: float = 0.01,
        k: int = 5,
        dataset_type: str = "binary",
    ):
        # alpha is beta from algorithm in the paper by Wang et al.
        super().__init__(hierarchy, alpha=alpha, dataset_type=dataset_type)
        self.k = k

    def _hill_climb(self, X) -> list[int]:
        self._score_matrix = self._calculate_scores(X)

        current_feature_set = get_leaves(self._feature_tree)
        if current_feature_set == ["ROOT"]:
            return []
        current_fitness = self._fitness_function(
            self._calculate_distances(current_feature_set)
        )

        unvisited = set(current_feature_set)

        while unvisited:
            temporary_feature_set = current_feature_set.copy()
            node = unvisited.pop()
            parent = list(self._feature_tree.predecessors(node))[0]
            temporary_feature_set.append(parent)
            children = list(self._feature_tree.successors(parent))
            updated_feature_set = [
                node for node in temporary_feature_set if node not in children
            ]
            temporary_fitness = self._fitness_function(
                self._calculate_distances(updated_feature_set)
            )
            if temporary_fitness < current_fitness:
                current_feature_set = temporary_feature_set
                current_fitness = temporary_fitness
                unvisited = set(current_feature_set)

        return current_feature_set

    def _calculate_distance(self, sample_i: int, sample_j: int, feature_set: list[int]):
        if "ROOT" in feature_set:
            feature_set.remove("ROOT")
        row_i = self._score_matrix[sample_i, feature_set]
        row_j = self._score_matrix[sample_j, feature_set]
        return cosine_similarity(row_i, row_j)

    def _fitness_function(self, distances: np.ndarray) -> float:
        number_of_leaf_nodes = len(get_leaves(self._feature_tree))  # alpha from paper

        threshold_index = self.n_features_ - self.k - 1
        k_nearest_neigbors = [
            list(np.argpartition(distances[row, :], threshold_index)[threshold_index:])
            for row in range(self._num_rows)
        ]

        count = 0
        for row in range(self._num_rows):
            for neighbor in k_nearest_neigbors[row]:
                if self.y_[neighbor] == self.y_[row] and neighbor != row:
                    count += 1

        result = count * (
            1
            + self.alpha
            * (number_of_leaf_nodes - self.n_features_)
            / number_of_leaf_nodes
        )
        return result
