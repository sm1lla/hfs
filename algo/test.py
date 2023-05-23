from fractions import Fraction
from sklearn.utils.estimator_checks import check_estimator
from hnb import HNB
from fixtures import getFixedData, getFixedDag
from helpers import getRelevance
from filter import Filter
import networkx as nx
import numpy as np

#Fixtures
bigDAG = getFixedDag()
data = getFixedData(20)

small_DAG = nx.to_numpy_array(nx.DiGraph([(0,1),(0,2),(1,2),(1,3)]))
small_x_data = np.array([[1,1,0,1],[1,0,0,0], [1,1,1,0],[1,1,1,1]])
small_y_data = np.array([0, 0, 1, 1])

# Test scikit-compatibility
sest = check_estimator(Filter())
assert(sest == None)

# Test Relevance calculation
results = [Fraction(1,2),Fraction(8,9),2,0]
for node_idx in range(len(small_DAG)):
    value = getRelevance(small_x_data, small_y_data, node_idx)
    assert(value == results[node_idx])
    

x = HNB(getFixedDag())
x_t, y_t = getFixedData(5)
x_test, y_test = getFixedData(5)    
x.fit_selector(x_t, y_t, x_test)




