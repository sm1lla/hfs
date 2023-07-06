import json
import networkx as nx
import pandas as pd

from hfs.preprocessing import HierarchicalPreprocessor


from hfs.hnb import HNB


def data():
    graph = nx.read_gml("../hfs/data/go_digraph2.gml")
    df = pd.read_csv("../hfs/data/train_test.csv")
    train = df[df["split"]=="train"]
    test = df[df["split"]=="test"]
    g2g = pd.read_csv(r"../hfs/data/gene2gomod3.txt",sep=",")
    go_terms = g2g["GO_ID"].unique()
    y_train = train["longevity influence"].to_numpy()
    y_test = test["longevity influence"].to_numpy()
    columns = []
    for node in train.columns:
        columns.append(list(graph.nodes()).index(node) if node in graph.nodes else -1)
    train = train[go_terms].to_numpy()
    test = test[go_terms].to_numpy()
    hierarchy = nx.to_numpy_array(graph)
    return (hierarchy, train, y_train, test, y_test, columns)


# Evalueate feature selection of HNB
def evaluate_HNB(data):
    hierarchy, train, y_train, test, y_test, columns = data()
    preprocessor = HierarchicalPreprocessor(hierarchy=hierarchy)
    preprocessor.fit(train, columns=columns)
    train = preprocessor.transform(train)
    test = preprocessor.transform(test)
    hierarchy = preprocessor.get_hierarchy()
    for 
    filter = HNB(hierarchy=hierarchy, k=2)
    filter.fit_selector(X_train=train, y_train=y_train, X_test=test)
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    score = filter.get_score(y_test, pred)
    with open('..//hfs/hfs/results/hnb.txt', 'w') as file:
        file.write(json.dumps(score))
def a():
    score = {"hi":0}
    with open('../hfs/results/hnb.txt', 'w') as file:
        file.write(json.dumps(score))

a()
evaluate_HNB(data)