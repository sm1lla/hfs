from sklearn.utils.estimator_checks import check_estimator
from fixtures import getFixedData, getFixedDag
from helpers import getRelevance
from filter import Filter
import networkx as nx
import numpy as np

#Fixtures
bigDAG = getFixedDag()
data = getFixedData(20)

smallDAG = nx.to_numpy_array(nx.DiGraph([(0,1),(0,2),(1,2),(1,3)]))
smallData = np.array([[1,1,0,1],[1,0,0,0], [1,1,1,0],[1,1,1,1]])

# Test scikit-compatibility
sest = check_estimator(Filter())
print(sest)

# Test Relevance calculation
#for node in smallDAG:
#    getRelevance(smallDAG, smallData, node)



