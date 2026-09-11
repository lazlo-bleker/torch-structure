import torch

from torch_structure.data import StructData


def test_laplacian_coordinates_and_cache():
    data = StructData(
        edge_index=torch.tensor([[0, 1, 1, 0], [1, 0, 2, 2]]),
        coords=torch.tensor([[0.0], [2.0], [4.0]]),
    )

    expected = torch.tensor([[-2.0], [2.0], [3.0]])

    assert torch.allclose(data.laplacian_coordinates(), expected)


def test_laplacian_cache_refreshes_when_topology_changes():
    data = StructData(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        coords=torch.tensor([[0.0], [2.0]]),
    )
    old_laplacian = data.laplacian

    data.edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 2, 2]])
    data.num_nodes = 3

    assert data.laplacian is not old_laplacian
