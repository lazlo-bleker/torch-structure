import pytest
import torch
from torch_structure.data.data import StructData

@pytest.fixture
def test_add_nodes():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    coords = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])

    new_node_attrs = {
        "coords": coords
    }

    data.add_nodes(**new_node_attrs)

    num_nodes_expected = torch.tensor(3)
    coords_expected = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])

    assert torch.equal(torch.tensor(data.num_nodes), num_nodes_expected)
    assert torch.equal(data.coords, coords_expected)



