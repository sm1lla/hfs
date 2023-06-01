import numpy as np
import networkx as nx
from fixtures import getFixedData, getFixedDag


#Fixtures
bigDAG = getFixedDag()
data = getFixedData(20)

small_DAG = nx.to_numpy_array(nx.DiGraph([(0,1),(0,2),(1,2),(1,3)]))
train_x_data = np.array([[1,1,0,1],[1,0,0,0], [1,1,1,0],[1,1,1,1]])
train_y_data = np.array([0, 0, 1, 1])
test_x_data = np.array([[1,1,0,0], [1,1,1,0]])
test_y_data = np.array([1,0])