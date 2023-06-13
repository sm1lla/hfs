from fractions import Fraction

import numpy as np
import networkx as nx


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

def get_leaves(x_identifier, digraph):
    return [x for x in digraph.nodes() if digraph.out_degree(x)==0 and digraph.in_degree(x)==1 and x not in x_identifier]

def shrink_dag(x_identifier, digraph):
    leaves = get_leaves(x_identifier=x_identifier, digraph=digraph)
    while(leaves):
        for x in leaves:
            digraph.remove_node(x)
        leaves = get_leaves(x_identifier=x_identifier, digraph=digraph)
    return digraph

def connect_dag(x_identifiers, digraph):

    top_sort = nx.topological_sort(digraph)

    # connect every node with at least one ancestor on each path that is for shure in x_i
    # i = 0: source is either in or not in, as they are no predecessors,
    # there should not be any artificial edge
    # i: for each pred there is a direct edge to the pred and iff pred not in x_ide
    #       also to their pred2. (it does not matter if pred2 is really in x, if it is not, 
    #       the edge will be removed later anyway)
    # i+1: if i is in -> no artificial edge on this path needed
    #       if i is not -> artifical edge to every pred of i, so each path going through i
    #       will be continued, if i is removed later
    
    new_graph = digraph.copy()

    for node in list(top_sort):
        preds = list(digraph.predecessors(node))
        for pred in preds:
            new_connections = []
            if pred in x_identifiers: 
                #to do -> make more efficient
                #sort x_identifiers according to order in digraph
                #x_identifiers.sort(key = lambda i: top_sort.index())
                pass
            else:
                for pred_of_pred in digraph.predecessors(pred):
                    new_connections.append(pred_of_pred)
                for new_connection in new_connections:
                    digraph.add_edge(new_connection, node)

    # remove all nodes (and edges) that are not in x_identifier
    x_identifiers_set = set(x_identifiers)
    nodes_to_remove = [node for node in digraph.nodes if node not in x_identifiers_set]
    digraph.remove_nodes_from(nodes_to_remove)

    return digraph
