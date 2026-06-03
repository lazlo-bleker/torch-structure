import pytest
import torch
from torch_structure.data.data import StructData

@pytest.fixture
def test_add_n_empty_nodes():

    data = StructData()

    data.add_n_empty_nodes(17)

    num_nodes_expected = torch.tensor(17)

    assert torch.equal(torch.tensor(data.num_nodes), num_nodes_expected)


def test_add_n_empty_nodes_with_non_empty_node_attribute_list():

    print("test_add_n_empty_nodes_with_non_empty_node_attribute_list not implemented yet")