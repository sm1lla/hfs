import json
import networkx as nx
import pandas as pd
from sklearn.model_selection import KFold
import math

from hfs.preprocessing import HierarchicalPreprocessor
from hfs.hnb import HNB
from hfs.hnbs import HNBs
from hfs.rnb import RNB
from hfs.hip import HIP
from hfs.mr import MR
from hfs.tan import Tan


models = [
    HNB(),
    HNBs(),
    RNB(),
    MR(),
    HIP(),
    Tan(),
]

def evaluate_model(data):
    hierarchy, X, y, columns = data()
    preprocessor = HierarchicalPreprocessor(hierarchy=hierarchy)
    preprocessor.fit(train, columns=columns)
    X = preprocessor.transform(X)
    hierarchy = preprocessor.get_hierarchy()
    kf = KFold(n_splits=10)
    filter = HIP(hierarchy=hierarchy)
    accuracy = 0
    specivity = 0
    sensitivity = 0
    for model in models:
        xtrain = X[train]
        xtest = X[test]
        ytrain = y[train]
        ytest = y[test]
        filter.fit_selector(X_train=xtrain, y_train=ytrain, X_test=xtest)
        pred = filter.select_and_predict(predict=True, saveFeatures=True)
        score = filter.get_score(ytest, pred)
        accuracy += score["accuracy"]
        specivity += score["0"]["recall"] 
        sensitivity += score["1"]["recall"]
    gmean = math.sqrt(sensitivity*specificity)
    score = {"accuracy": accuracy, "specificity": specificity, "sensitivity": sensitivity, "gmean":gmean}
    with open(f'../hfs/results/hip-10fold-comp.txt', 'w') as file:
        file.write(json.dumps(score))

