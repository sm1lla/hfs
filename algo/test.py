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
print(sest)

# Test Relevance calculation
for node in small_DAG:
    print(getRelevance(small_DAG, small_x_data, small_y_data, node))
    

x = HNBs(getFixedDag())
x_t, y_t = getFixedData(50)
x_test, y_test = getFixedData(5)    
x.fit_selector(x_t, y_t, x_test)
pred = x.select_and_predict(True, True)
print(x.score(y_test, pred))
fe = x.get_features()
#print(fe)

x_t, y_t = getFixedData(50)
x_test, y_test = getFixedData(5)
dag = nx.from_numpy_array(getFixedDag(), parallel_edges = False, create_using = nx.DiGraph)

checkData(dag, x_test, y_test)


