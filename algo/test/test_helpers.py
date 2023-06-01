import sys
sys.path.append('/home/kathrin/hfs/algo/')

from go import open_dag
import numpy as np
import networkx as nx

from helpers import connect_dag, shrink_dag
from test_fixtures import *
import matplotlib.pyplot as plt

def test_shrink_dag():

    nodes = np.load("./algo/data/nodes_go.npy")
    graph = open_dag("./algo/data/go_digraph")

    nonexist_nodes = ['GO:2001301', 'GO:2001302', 'GO:2001303', 'GO:2001304', 
                        'GO:2001305', 'GO:2001306', 'GO:2001307', 'GO:2001308', 
                        'GO:2001309', 'GO:2001310', 'GO:2001311', 'GO:2001312', 
                        'GO:2001313', 'GO:2001314', 'GO:2001315', 'GO:2001316',
                        'GO:2001092', 'GO:2001094', 'GO:2001106', 'GO:2001107',] #4 nodes that are leaves

    x_identifiers = np.setdiff1d(nodes, nonexist_nodes )

    # four nodes in non_existing_nodes can be removed as they are leaves
    assert(len([x for x in nonexist_nodes if graph.out_degree(x)==0 and graph.in_degree(x)==1])==4)

    # test removal of the nodes
    assert(len(graph.nodes())==43008)
    graph = shrink_dag(x_identifiers, graph)
    assert(len(graph.nodes())==43004)
    for node in ['GO:2001092', 'GO:2001094', 'GO:2001106', 'GO:2001107']:
        assert(not (node in graph.nodes()))

def test_connect_dag():
    graph = nx.DiGraph(getFixedDag())
    x_identifiers = [0, 1, 2, 5, 6, 7, 8]
    graph = connect_dag(digraph=graph, x_identifiers=x_identifiers)
    new_graph = nx.DiGraph([(0,1),(0,2),(1,6),(1,5),(1,7),(0,7),(5,8)])
    assert(nx.is_isomorphic(graph, new_graph))


    #nx.draw_networkx(graph)
    #plt.show()


test_connect_dag()
test_shrink_dag()