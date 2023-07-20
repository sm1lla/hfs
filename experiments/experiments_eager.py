import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from hfs.data.data_utils import create_mapping_columns_to_nodes, load_data
from hfs.gtd import GreedyTopDownSelector
from hfs.hill_climbing import BottomUpSelector, TopDownSelector
from hfs.preprocessing import HierarchicalPreprocessor
from hfs.shsel import SHSELSelector
from hfs.tsel import TSELSelector


def get_preprocessed_data():
    X, y, hierarchy = load_data()
    columns = create_mapping_columns_to_nodes(X, hierarchy)
    X = X.to_numpy()
    hierarchy = nx.to_numpy_array(hierarchy)
    preprocessor = HierarchicalPreprocessor(hierarchy)
    preprocessor.fit(X, columns)
    X_transformed = preprocessor.transform(X)
    hierarchy_updated = preprocessor.get_hierarchy()
    columns_updated = preprocessor.get_columns()
    return X_transformed, y, hierarchy_updated, columns_updated


def shsel(X, y, hierarchy, columns):
    print("SHSEL Feature Selection")
    selector = SHSELSelector(hierarchy)
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)
    return X_transformed


def hfe(X, y, hierarchy, columns):
    print("HFE Feature Selection")
    selector = SHSELSelector(
        hierarchy, relevance_metric="Correlation", use_hfe_extension=True
    )
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)
    return X_transformed


def tsel(X, y, hierarchy, columns):
    print("TSEL Feature Selection")
    selector = TSELSelector(hierarchy)
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)
    return X_transformed


def top_down(X, y, hierarchy, columns):
    print("Hill Climbing Top Down Feature Selection")
    selector = TopDownSelector(hierarchy)
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)
    return X_transformed


def bottom_up(X, y, hierarchy, columns):
    print("Hill Climbing Bottom Up Feature Selection")
    selector = BottomUpSelector(hierarchy)
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)
    return X_transformed


def gtd(X, y, hierarchy, columns):
    print("Greedy Top Down Feature Selection")
    selector = GreedyTopDownSelector(hierarchy)
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)
    return X_transformed


def classify(X, y, classifier):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    classifier.fit(X_train, y_train)
    accuracy = classifier.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")


def classification_experiments():
    experiments = [shsel, hfe, tsel, gtd, bottom_up, top_down]

    classifier = GaussianNB()
    print("Baseline without feature extraction")
    X, y, hierarchy, columns = get_preprocessed_data()
    classify(X, y, classifier)

    for experiment in experiments:
        X_transformed = experiment(X, y, hierarchy, columns)
        classify(X_transformed, y, classifier)
