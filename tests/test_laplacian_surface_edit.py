import torch

from torch_structure.data import StructData


def test_laplacian_surface_edit_preserves_supports_and_handle():
    data = StructData(
        edge_index=torch.tensor(
            [[0, 1, 1, 0, 1, 2, 2, 1], [1, 0, 2, 2, 0, 1, 1, 0]]
        ),
        coords=torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        is_support=torch.tensor([[True], [False], [False]]),
    )

    edited = data.laplacian_surface_edit(
        node_indices=torch.tensor([2]),
        displacements=torch.tensor([[0.0, 1.0]]),
    )

    assert torch.equal(edited[0], data.coords[0])
    assert torch.equal(edited[2], data.coords[2] + torch.tensor([0.0, 1.0]))
    assert edited.shape == data.coords.shape


def test_laplacian_surface_edit_supports_soft_handles():
    data = StructData(
        edge_index=torch.tensor(
            [[0, 1, 1, 0, 1, 2, 2, 1], [1, 0, 2, 2, 0, 1, 1, 0]]
        ),
        coords=torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        is_support=torch.tensor([[True], [False], [False]]),
    )

    edited = data.laplacian_surface_edit(
        node_indices=[2],
        displacements=[[0.0, 1.0]],
        handle_weight=1000.0,
    )

    assert torch.equal(edited[0], data.coords[0])
    assert edited[2, 1] > 0.99
