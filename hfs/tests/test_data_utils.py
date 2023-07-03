from ..data.data_utils import create_mapping_columns_to_nodes, load_data, process_data


def test_load_data():
    X, labels, hierarchy = load_data()
    mapping = create_mapping_columns_to_nodes(X, hierarchy)
    assert len(mapping) == 269
    assert len(X) == labels.shape[0]
