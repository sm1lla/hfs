import json
import pathlib

import networkx as nx
import pandas as pd

from hfs.hip import HIP
from hfs.hnb import HNB
from hfs.hnbs import HNBs
from hfs.mr import MR
from hfs.preprocessing import HierarchicalPreprocessor
from hfs.rnb import RNB
from hfs.tan import Tan


def data():
    dir = pathlib.Path(__file__).parent.parent.absolute()
    rel = pathlib.Path("hfs/data/go_digraph2.gml")
    path = dir / rel
    graph = nx.read_gml(path)
    rel = pathlib.Path("hfs/data/train_test.csv")
    path = dir / rel
    df = pd.read_csv(path)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    rel = pathlib.Path("dhfs/ata/gene2gomod3.txt")
    path = dir / rel
    g2g = pd.read_csv(path, sep=",")
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


def hnb(hierarchy, train, y_train, test, y_test, k):
    model = HNB(hierarchy=hierarchy, k=k)
    model.fit_selector(X_train=train, y_train=y_train, X_test=test)
    pred = model.select_and_predict(predict=True, saveFeatures=True)
    score = model.get_score(y_test, pred)
    rel = pathlib.Path(f"results/all.txt")
    path = dir / rel
    with open(path, "a") as file:
        file.write("HNB:\n")
        file.write(json.dumps(score))


def hnbs(hierarchy, train, y_train, test, y_test, k):
    model = HNBs(hierarchy=hierarchy)
    model.fit_selector(X_train=train, y_train=y_train, X_test=test)
    pred = model.select_and_predict(predict=True, saveFeatures=True)
    score = model.get_score(y_test, pred)
    rel = pathlib.Path(f"results/all.txt")
    path = dir / rel
    with open(path, "a") as file:
        file.write("HNBs:\n")
        file.write(json.dumps(score))


def rnb(hierarchy, train, y_train, test, y_test, k):
    model = RNB(hierarchy=hierarchy, k=k)
    model.fit_selector(X_train=train, y_train=y_train, X_test=test)
    pred = model.select_and_predict(predict=True, saveFeatures=True)
    score = model.get_score(y_test, pred)
    rel = pathlib.Path(f"results/all.txt")
    path = dir / rel
    with open(path, "a") as file:
        file.write("RNB:\n")
        file.write(json.dumps(score))


def mr(hierarchy, train, y_train, test, y_test, k):
    model = MR(hierarchy=hierarchy)
    model.fit_selector(X_train=train, y_train=y_train, X_test=test)
    pred = model.select_and_predict(predict=True, saveFeatures=True)
    score = model.get_score(y_test, pred)
    rel = pathlib.Path(f"results/all.txt")
    path = dir / rel
    with open(path, "a") as file:
        file.write("MR:\n")
        file.write(json.dumps(score))


def tan(hierarchy, train, y_train, test, y_test, k):
    model = Tan(hierarchy=hierarchy)
    model.fit_selector(X_train=train, y_train=y_train, X_test=test)
    pred = model.select_and_predict(predict=True, saveFeatures=True)
    score = model.get_score(y_test, pred)
    rel = pathlib.Path(f"results/all.txt")
    path = dir / rel
    with open(path, "a") as file:
        file.write("Tan:\n")
        file.write(json.dumps(score))


def hip(hierarchy, train, y_train, test, y_test, k):
    model = HIP(hierarchy=hierarchy)
    model.fit_selector(X_train=train, y_train=y_train, X_test=test)
    pred = model.select_and_predict(predict=True, saveFeatures=True)
    score = model.get_score(y_test, pred)
    rel = pathlib.Path(f"results/all.txt")
    path = dir / rel
    with open(path, "a") as file:
        file.write("HIP:\n")
        file.write(json.dumps(score))


# Evalueate feature selection of HNB
def evaluate(data, k):
    hierarchy, train, y_train, test, y_test, columns = data()
    preprocessor = HierarchicalPreprocessor(hierarchy=hierarchy)
    preprocessor.fit(train, columns=columns)
    train = preprocessor.transform(train)
    test = preprocessor.transform(test)
    hierarchy = preprocessor.get_hierarchy()
    for function in [hnb, hnbs, rnb, tan, mr, hip]:
        function(
            hierarchy=hierarchy,
            train=train,
            y_train=y_train,
            test=test,
            y_test=y_test,
            k=k,
        )


evaluate(data, 20)
