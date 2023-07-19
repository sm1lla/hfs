import pandas as pd
import pytest

from ..data.data_utils import create_mapping_columns_to_nodes, load_data, process_data
from .fixtures.fixtures import (
    dataframe,
    hierarchy1,
    hierarchy1_2,
    hierarchy2,
    hierarchy3,
)


def test_load_data():
    X, labels, hierarchy = load_data()
    mapping = create_mapping_columns_to_nodes(X, hierarchy)
    assert len(mapping) == 269
    assert len(X) == labels.shape[0]


@pytest.mark.parametrize(
    "hierarchy, dataframe",
    [
        (hierarchy1(), dataframe()),
        (hierarchy1_2(), dataframe()),
        (hierarchy2(), dataframe()),
        (hierarchy3(), dataframe()),
    ],
)
def test_create_mapping_columns_to_nodes(hierarchy, dataframe):
    mapping = create_mapping_columns_to_nodes(dataframe, hierarchy)
    nodes = list(hierarchy.nodes)
    for index, node in enumerate(dataframe.columns):
        assert nodes[mapping[index]] == node
