from fractions import Fraction
from sklearn.utils.estimator_checks import check_estimator
from hnb import HNB, HNBs, RNB
from fixtures import getFixedData, getFixedDag
from helpers import getRelevance
from filter import Filter
import networkx as nx
import numpy as np

#Fixtures
bigDAG = getFixedDag()
data = getFixedData(20)

small_DAG = nx.to_numpy_array(nx.DiGraph([(0,1),(0,2),(1,2),(1,3)]))
train_x_data = np.array([[1,1,0,1],[1,0,0,0], [1,1,1,0],[1,1,1,1]])
train_y_data = np.array([0, 0, 1, 1])
test_x_data = np.array([[1,1,0,0], [1,1,1,0]])
test_y_data = np.array([1,1])

# Test scikit-compatibility
sest = check_estimator(Filter())
assert(sest == None)

# Test Relevance calculation
results = [Fraction(1,2), Fraction(8,9), 2, 0]
for node_idx in range(len(small_DAG)):
    value = getRelevance(train_x_data, train_y_data, node_idx)
    assert(value == results[node_idx])

# Test feature selection of HNB
filter = HNB(graph_data=small_DAG, k=2)
filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
pred = filter.select_and_predict(predict=True, saveFeatures=True)
print(f"hnbspred: {pred}")

print(f"features: {filter.get_features()}\n")

# Test feature selection of HNBs
filter = HNBs(graph_data=small_DAG)
filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
pred = filter.select_and_predict(predict=True, saveFeatures=True)
print(f"hnbsfeatures: {filter.get_features()}")
print(f"pred: {pred}")
print(f"score: {filter.get_score(test_y_data, pred)}\n")

# Test feature selection of RNB
filter = RNB(graph_data=small_DAG, k=2)
filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
pred = filter.select_and_predict(predict=True, saveFeatures=True)
print(f"rnbfeatures: {filter.get_features()}")
print(f"pred: {pred}")
print(f"score: {filter.get_score(test_y_data, pred)}\n")