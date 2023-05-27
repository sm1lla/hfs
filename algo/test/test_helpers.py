import sys
sys.path.append('/home/kathrin/hfs/algo/')
from data.go import open_dag
import nxontology
import networkx as nx
import numpy as np

from helpers import get_leaves, shrink_dag

def test_shrink_dag():
    x_identifiers = ""
    
    nodes = np.load("./algo/data/nodes_go.npy")
    graph = open_dag("./algo/data/go_digraph2")

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

test_shrink_dag()