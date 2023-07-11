import json
import pathlib
import networkx as nx
import pandas as pd

from ..preprocessing import HierarchicalPreprocessor


from ..hip import HIP

from ..preprocessing import HierarchicalPreprocessor
from ..hnb import HNB
from ..hnbs import HNBs
from ..rnb import RNB
from ..hip import HIP
from ..mr import MR
from ..tan import Tan


models = [
    HNB(),
    HNBs(),
    RNB(),
    MR(),
    HIP(),
    Tan(),
]
model_names = [
    "HNB",
    "HNBs",
    "RNB",
    "MR",
    "HIP",
    "Tan",
]
def data():
    dir = pathlib.Path(__file__).parent.parent.absolute()
    rel = pathlib.Path('data/go_digraph2.gml')
    path = dir /rel 
    graph = nx.read_gml(path)
    rel = pathlib.Path("data/train_test.csv")
    path = dir /rel 
    df = pd.read_csv(path)
    train = df[df["split"]=="train"]
    test = df[df["split"]=="test"]
    rel = pathlib.Path("data/gene2gomod3.txt")
    path = dir /rel 
    g2g = pd.read_csv(path,sep=",")
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
def evaluate(data):
    hierarchy, train, y_train, test, y_test, columns = data()
    preprocessor = HierarchicalPreprocessor(hierarchy=hierarchy)
    preprocessor.fit(train, columns=columns)
    train = preprocessor.transform(train)
    test = preprocessor.transform(test)
    hierarchy = preprocessor.get_hierarchy()
    model_idx = 0
    for model in models:
        filter = model(hierarchy=hierarchy)
        filter.fit_selector(X_train=train, y_train=y_train, X_test=test)
        pred = filter.select_and_predict(predict=True, saveFeatures=True)
        score = filter.get_score(y_test, pred)
        rel = pathlib.Path(f'results/{model_names[model_idx]}.txt')
        path = dir /rel 
        with open(path, 'w') as file:
            file.write(json.dumps(score))

evaluate(data)