import torch
from torch_structure.data.data import StructData


def test_polar_grid_structure():

    node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
    edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    def coords(r_ind, a_ind):
        return torch.hstack([r_ind, a_ind, torch.zeros_like(r_ind)])

    def force(x_unit_coord):
        return x_unit_coord

    data.add_polar_grid(3, 2, node_attrs={"coords": coords}, edge_attrs={"force": force})

    edge_index_expected = torch.tensor([
        [0, 0, 0, 1, 3, 5, 1, 2, 3, 4, 5, 6, 1, 3, 5, 2, 4, 6, 3, 4, 5, 6, 1, 2],
        [1, 3, 5, 2, 4, 6, 3, 4, 5, 6, 1, 2, 0, 0, 0, 1, 3, 5, 1, 2, 3, 4, 5, 6],
    ], dtype=torch.long)
    directed_mask_expected = torch.tensor([True] * 12 + [False] * 12).unsqueeze(-1)
    reciprocal_edge_expected = torch.tensor([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).unsqueeze(-1)
    num_nodes_expected = torch.tensor(7)
    coords_expected = torch.tensor([
        [0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
        [1., 1., 0.], [2., 1., 0.], [1., 2., 0.], [2., 2., 0.],
    ])
    force_expected = torch.tensor([
        [ 0.2500], [-0.1250], [-0.1250], [ 0.7500], [-0.3750], [-0.3750],
        [ 0.1250], [ 0.2500], [-0.2500], [-0.5000], [ 0.1250], [ 0.2500],
        [ 0.2500], [-0.1250], [-0.1250], [ 0.7500], [-0.3750], [-0.3750],
        [ 0.1250], [ 0.2500], [-0.2500], [-0.5000], [ 0.1250], [ 0.2500],
    ])

    assert torch.equal(data.reciprocal_edge, reciprocal_edge_expected)
    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.directed_mask, directed_mask_expected)
    assert torch.equal(torch.tensor(data.num_nodes), num_nodes_expected)
    assert torch.equal(data.coords, coords_expected)
    assert torch.allclose(data.force, force_expected)
