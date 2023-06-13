
from nxontology import NXOntology
from nxontology.imports import from_file, pronto_to_multidigraph, multidigraph_to_digraph
import networkx as nx
import pronto
import numpy as np

def load_dag(url, file_name):
    nxo = from_file(url)
    nxo.n_nodes
    nx.number_weakly_connected_components(nxo.graph)

    go_pronto = pronto.Ontology(handle=url)
    go_multidigraph = pronto_to_multidigraph(go_pronto)
    go_digraph = multidigraph_to_digraph(go_multidigraph, reduce=True)
    
    go_digraph_sparse = nx.to_scipy_sparse_array(go_digraph)
    nx.write_gml(go_digraph, file_name+".gml")
    l = go_digraph.nodes()
    np.save("./algo/data/nodes_go.npy", l)


def open_dag(file_name):
    graph = nx.read_gml(file_name+".gml")
    

    return graph

def main():
    url = "http://release.geneontology.org/2023-05-10/ontology/go-basic.json.gz"
    file_name = "./algo/data/go_digraph"
    load_dag(url=url, file_name=file_name)
    open_dag(file_name=file_name)
