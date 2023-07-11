import json
import networkx as nx
import pandas as pd

from hfs.preprocessing import HierarchicalPreprocessor


from hfs.hnb import HNB
from hfs.tests.fixtures.fixtures import *


def data():
    graph = nx.read_gml("../hfs/data/go_digraph2.gml")
    subgraph = graph.subgraph([
        "GO:2001301",
        "GO:2001302",
        "GO:2001303",
        "GO:2001304",
        "GO:2001305",
        "GO:2001306",
        "GO:2001307",
        "GO:2001308",
        "GO:2001309",
        "GO:2001310",
        "GO:2001311",
        "GO:2001312",
        "GO:2001313",
        "GO:2001314",
        "GO:2001315",
        "GO:2001316",
        "GO:2001092",
        "GO:2001094",
        "GO:2001106",
        "GO:2001107",
    ] )
    df = pd.read_csv("../hfs/data/train_test.csv")
    train = df[df["split"]=="train"]
    test = df[df["split"]=="test"]
    g2g = pd.read_csv(r"../hfs/data/gene2gomod3.txt",sep=",")
    go_terms = g2g["GO_ID"].unique()
    y_train = train["longevity influence"].to_numpy()
    y_test = test["longevity influence"].to_numpy()
    columns = []
    for node in train.columns:
        columns.append(list(subgraph.nodes()).index(node) if node in subgraph.nodes else -1)
    train = train[go_terms].to_numpy()
    test = test[go_terms].to_numpy()
    hierarchy = nx.to_numpy_array(subgraph)
    return (hierarchy, train, y_train, test, y_test, columns)


def test_preprocessing(data):
    hierarchy, train, y_train, test, y_test, columns = data()
    preprocessor = HierarchicalPreprocessor(hierarchy=hierarchy)
    preprocessor.fit(train, columns=columns)
    train_new = preprocessor.transform(train)
    print(train_new)
    return (hierarchy, train, test, y_train, y_test)

# Test feature selection of HNB
def test_HNB(data):
    hierarchy, train, y_train, test, y_test, columns = data()
    preprocessor = HierarchicalPreprocessor(hierarchy=hierarchy)
    preprocessor.fit(train, columns=columns)
    train = preprocessor.transform(train)
    test = preprocessor.transform(test)
    hierarchy = preprocessor.get_hierarchy()
    filter = HNB(hierarchy=hierarchy, k=2)
    filter.fit_selector(X_train=train, y_train=y_train, X_test=test)
    pred = filter.select_and_predict(predict=True, saveFeatures=True)
    

hierarchy, train, y_train, test, y_test, columns = data()
preprocessor = HierarchicalPreprocessor(hierarchy=hierarchy)
preprocessor.fit(train, columns=columns)
train = preprocessor.transform(train)
test = preprocessor.transform(test) 
hierarchy = preprocessor.get_hierarchy()
filter = HNB(hierarchy=hierarchy, k=10)
filter.fit_selector(X_train=train, y_train=y_train, X_test=test)
pred = filter.select_and_predict(predict=True, saveFeatures=True)

score = filter.get_score(y_test, pred)

with open('../hfs/results/hnbex.txt', 'w') as file:
    file.write(json.dumps(score))