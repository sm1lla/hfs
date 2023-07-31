import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from kgextension.generator import direct_type_generator


def process_data(
    path: str = "hfs/data/sport_tweets_train.tsv", test_version: bool = False
):
    """Extends data with types and build hierarchy. Saves dataframe and hierarchy graph to a csv file"""
    data = pd.read_csv(Path(path), sep="\t")
    if test_version:
        data = data[:10]
    else:
        data = data[:600]
    extended_data = direct_type_generator(
        data,
        ["Dbpedia_URI_1", "Dbpedia_URI_2", "Dbpedia_URI_3"],
        hierarchy=True,
        caching=False,
    )
    graph = extended_data.attrs["hierarchy"]
    cleaned_data = extended_data[
        [column for column in extended_data.columns if "type" in column]
    ]
    column_names_mapping = dict(
        zip(
            cleaned_data.columns,
            [column.split("_")[1] for column in cleaned_data.columns],
        )
    )
    cleaned_data.rename(column_names_mapping, inplace=True, axis=1)
    cleaned_data = cleaned_data.astype(float)
    cleaned_data["label"] = data["label"]

    version = ""
    if test_version:
        version = "_testing"
    else:
        version = "_subset600"
    pickle.dump(
        graph, open(Path(f"{path.split('.')[0]}_hierarchy{version}.pickle"), "wb")
    )
    cleaned_data.to_csv(
        Path(f"{path.split('.')[0]}_with_hierarchy{version}.csv"), index=False
    )


def load_data(
    path: str = "hfs/data/sport_tweets_train.tsv", test_version: bool = False
) -> tuple[pd.DataFrame, nx.DiGraph, np.ndarray]:
    version = ""
    if test_version:
        version = "_testing"
    else:
        version = "_subset300"
    hierarchy = pickle.load(
        open(Path(f"{path.split('.')[0]}_hierarchy{version}.pickle"), "rb")
    )
    data = pd.read_csv(Path(f"{path.split('.')[0]}_with_hierarchy{version}.csv"))
    labels = np.array(data["label"])
    data.drop("label", axis=1, inplace=True)
    return data, labels, hierarchy


def create_mapping_columns_to_nodes(data: pd.DataFrame, hierarchy: nx.DiGraph):
    columns = list(data.columns)
    nodes = list(hierarchy.nodes)
    mapping = [nodes.index(node) if node in nodes else -1 for node in columns]
    return mapping
