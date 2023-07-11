from sklearn.model_selection import KFold
import networkx as nx
import pandas as pd
def data():
    graph = nx.read_gml("../hfs/data/go_digraph2.gml")
    df = pd.read_csv("../hfs/data/dropped_genes_without_goandli.csv")
    g2g = pd.read_csv(r"../hfs/data/gene2gomod3.txt",sep=",")
    go_terms = g2g["GO_ID"].unique()
    columns = []
    for node in df.columns:
        columns.append(list(graph.nodes()).index(node) if node in graph.nodes else -1)
    X = df[go_terms].to_numpy()
    y = df["longevity influence"].to_numpy()
    hierarchy = nx.to_numpy_array(graph)

    return (hierarchy, X, y, columns)