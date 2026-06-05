import torch
from torch_structure.data.data import StructData


def test_triangular_grid_structure():

    node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
    edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    def coords(u_ind, v_ind):
        return torch.hstack([u_ind, v_ind, torch.zeros_like(u_ind)])

    def force(x_unit_coord):
        return x_unit_coord

    data.add_triangular_grid(3, node_attrs={"coords": coords}, edge_attrs={"force": force})

    edge_index_expected = torch.tensor([
        [0, 1, 3, 0, 1, 3, 1, 2, 4, 1, 2, 4, 3, 4, 5, 3, 4, 5],
        [1, 2, 4, 3, 4, 5, 3, 4, 5, 0, 1, 3, 0, 1, 3, 1, 2, 4],
    ], dtype=torch.long)
    directed_mask_expected = torch.tensor([True] * 9 + [False] * 9).unsqueeze(-1)
    reciprocal_edge_expected = torch.tensor([9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8]).unsqueeze(-1)
    num_nodes_expected = torch.tensor(6)
    coords_expected = torch.tensor([
        [0., 0., 0.], [0., 1., 0.], [0., 2., 0.],
        [1., 0., 0.], [1., 1., 0.], [2., 0., 0.],
    ])
    force_expected = torch.tensor([
        [0.00], [0.00], [0.50], [0.25], [0.25], [0.75], [0.25], [0.25], [0.75],
        [0.00], [0.00], [0.50], [0.25], [0.25], [0.75], [0.25], [0.25], [0.75],
    ])

    assert torch.equal(data.reciprocal_edge, reciprocal_edge_expected)
    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.directed_mask, directed_mask_expected)
    assert torch.equal(torch.tensor(data.num_nodes), num_nodes_expected)
    assert torch.equal(data.coords, coords_expected)
    assert torch.allclose(data.force, force_expected)
