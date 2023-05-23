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
    p2 = Fraction(xdata[(xdata[:,node]==1)& (ydata==0)].shape[0], xdata[(xdata[:,node]==1)].shape[0]) if xdata[(xdata[:,node]==1)].shape[0] != 0 else 0
    p3 = 1 -p1
    p4 = 1 -p2


    rel = (p1-p2)**2 + (p3-p4)**2
    #print(p1, p2, p3, p4, rel)
    return rel

def checkData(dag, x_data, y_data):
    data = np.column_stack((x_data, y_data))
    edges = list(nx.edge_dfs(dag, source=0, orientation = "original"))
    print(data)
    for edge in edges:
        for idx in range(len(data)):
            if data[idx,edge[0]] == 0 and data[idx,edge[1]] == 1:
                # depending on number of errors -> delete those instances instead of throwing errors? 
                raise ValueError(
                    f"Test instance {idx} violates 0-1-propagation on edge ({edge[0]}, {edge[1]})"
                    f"{data[idx]}"
                )
            