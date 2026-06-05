import torch
from torch_structure.data.data import StructData


def test_chain_structure():

    node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
    edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    def coords(u_ind):
        return torch.hstack([u_ind, torch.zeros_like(u_ind), torch.zeros_like(u_ind)])

    def force(x_unit_coord):
        return x_unit_coord

    data.add_chain(3, node_attrs={"coords": coords}, edge_attrs={"force": force})

    edge_index_expected = torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]], dtype=torch.long)
    directed_mask_expected = torch.tensor([True, True, False, False]).unsqueeze(-1)
    reciprocal_edge_expected = torch.tensor([2, 3, 0, 1]).unsqueeze(-1)
    num_nodes_expected = torch.tensor(3)
    coords_expected = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    force_expected = torch.tensor([[0.25], [0.75], [0.25], [0.75]])

    assert torch.equal(data.reciprocal_edge, reciprocal_edge_expected)
    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.directed_mask, directed_mask_expected)
    assert torch.equal(torch.tensor(data.num_nodes), num_nodes_expected)
    assert torch.equal(data.coords, coords_expected)
    assert torch.allclose(data.force, force_expected)
