from fractions import Fraction

import numpy as np
import networkx as nx

from fixtures import getFixedDag, getFixedData

    

def getRelevance(xdata, ydata, node):
    """
    Gather relevance for a given node.

    Parameters
    ----------
    node
        Node for which the relevance should be obtained.
    xdata
        xdata
    ydata
        data as np array
    """
    p1 = Fraction(xdata[(xdata[:,node]==1)& (ydata==1)].shape[0], xdata[(xdata[:,node]==1)].shape[0]) if xdata[(xdata[:,node]==1)].shape[0] != 0 else 0
    p2 = Fraction(xdata[(xdata[:,node]==0)& (ydata==1)].shape[0], xdata[(xdata[:,node]==0)].shape[0]) if xdata[(xdata[:,node]==0)].shape[0] != 0 else 0
    p3 = 1 -p1
    p4 = 1 -p2


    rel = (p1-p2)**2 + (p3-p4)**2
    return rel

def checkData(dag, x_data, y_data): # possible: bool checking
    data = np.column_stack((x_data, y_data))
    edges = list(nx.edge_dfs(dag, source=0, orientation = "original"))
    for edge in edges:
        for idx in range(len(data)):
            if data[idx,edge[0]] == 0 and data[idx,edge[1]] == 1:
                # depending on number of errors -> delete those instances instead of throwing errors? 
                raise ValueError(
                    f"Test instance {idx} violates 0-1-propagation on edge ({edge[0]}, {edge[1]})"
                    f"{data[idx]}"
                )
            

def expand_data(x_data, x_identifier, graph_data, graph_identifier):
    pass

def get_leaves(x_identifier, graph):
    return [x for x in graph.nodes() if graph.out_degree(x)==0 and graph.in_degree(x)==1 and x not in x_identifier]

def shrink_dag(x_identifier, graph):
    reversed_graph = graph.reverse()
    leaves = get_leaves(x_identifier=x_identifier, graph=graph)
    while(leaves):
        for x in leaves:
            graph.remove_node(x)
        leaves = get_leaves(x_identifier=x_identifier, graph=graph)
    return graph






    #sort x_identifiers
