import sys
sys.path.append('/home/kathrin/hfs/algo/')

from fractions import Fraction
from sklearn.utils.estimator_checks import check_estimator
from filter import Filter
from hnb import HNB, HNBs, RNB
from fixtures import getFixedData, getFixedDag
from helpers import getRelevance
#from algo.filter import Filter
import networkx as nx
import numpy as np
from test_fixtures import *

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
assert(np.array_equal(filter.get_features(),np.array([[0, 1, 1, 0], [0, 0, 1, 1]])))
assert(np.array_equal(pred, np.array([0, 1])))
assert(filter.get_score(test_y_data, pred) == 0.0)

# Test feature selection of HNBs
filter = HNBs(graph_data=small_DAG)
filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
pred = filter.select_and_predict(predict=True, saveFeatures=True)
assert(np.array_equal(pred,np.array([0, 1])))
assert(np.array_equal(filter.get_features(), np.array([[0, 1, 1, 1], [0, 0, 1, 1]])))
assert(filter.get_score(test_y_data, pred) == 0.0)

# Test feature selection of RNB
filter = RNB(graph_data=small_DAG, k=2)
filter.fit_selector(X_train=train_x_data, y_train=train_y_data, X_test=test_x_data)
pred = filter.select_and_predict(predict=True, saveFeatures=True)
assert(np.array_equal(pred,np.array([0, 1])))
assert(np.array_equal(filter.get_features(), np.array([[0, 1, 1, 0], [0, 1, 1, 0]])))
assert(filter.get_score(test_y_data, pred) == 0.0)


