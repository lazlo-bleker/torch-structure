import pytest
import torch
from torch_structure.data.data import StructData


def test_grid_with_only_node_attributes():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(
        node_attrs=node_attrs,
    )

    def coords(u_ind, v_ind):

        return torch.hstack([u_ind,v_ind,torch.zeros_like(u_ind)])


    node_attrs_grid = {
        "coords": coords
    }

    data.add_grid(2,2, node_attrs=node_attrs_grid)

    edge_index_expected = torch.tensor([[0, 2, 0, 1, 1, 3, 2, 3], [1, 3, 2, 3, 0, 2, 0, 1]], dtype=torch.long)
    directed_mask_expected = torch.tensor([True, True, True, True, False, False, False, False]).unsqueeze(-1)
    reciprocal_edge_expected = torch.tensor([4,5,6,7,0,1,2,3]).unsqueeze(-1)
    num_nodes_expected = torch.tensor(4) 
    coords_expected = torch.tensor([[0,0,0], [0,1,0], [1,0,0], [1,1,0]])

    assert torch.equal(data.reciprocal_edge, reciprocal_edge_expected)
    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.directed_mask, directed_mask_expected)
    assert torch.equal(torch.tensor(data.num_nodes), num_nodes_expected)
    assert torch.equal(getattr(data, "coords"), coords_expected)


def test_grid_with_only_edge_attributes():

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(
        edge_attrs=edge_attrs,
    )

    def force(u_ind):

        return torch.hstack([u_ind])

    edge_attrs_grid = {
        "force": force
    }

    data.add_grid(2,2, edge_attrs=edge_attrs_grid)

    edge_index_expected = torch.tensor([[0, 2, 0, 1, 1, 3, 2, 3], [1, 3, 2, 3, 0, 2, 0, 1]], dtype=torch.long)
    directed_mask_expected = torch.tensor([True, True, True, True, False, False, False, False]).unsqueeze(-1)
    reciprocal_edge_expected = torch.tensor([4,5,6,7,0,1,2,3]).unsqueeze(-1)
    num_nodes_expected = torch.tensor(4) 
    force_expected = torch.tensor([[0], [1], [0], [0], [0], [1], [0], [0]])

    assert torch.equal(data.reciprocal_edge, reciprocal_edge_expected)
    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.directed_mask, directed_mask_expected)
    assert torch.equal(torch.tensor(data.num_nodes), num_nodes_expected)
    assert torch.equal(getattr(data, "force"), force_expected)


def test_grid_with_node_and_edge_attributes():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(

        node_attrs=node_attrs,
        edge_attrs=edge_attrs

    )

    def force(u_ind):

        return torch.hstack([u_ind])

    def coords(u_ind, v_ind):

        return torch.hstack([u_ind,v_ind,torch.zeros_like(u_ind)])

    node_attrs_grid = {
        "coords": coords
    }

    edge_attrs_grid = {
        "force": force
    }

    data.add_grid(2,2, node_attrs=node_attrs_grid, edge_attrs=edge_attrs_grid)

    edge_index_expected = torch.tensor([[0, 2, 0, 1, 1, 3, 2, 3], [1, 3, 2, 3, 0, 2, 0, 1]], dtype=torch.long)
    directed_mask_expected = torch.tensor([True, True, True, True, False, False, False, False]).unsqueeze(-1)
    reciprocal_edge_expected = torch.tensor([4,5,6,7,0,1,2,3]).unsqueeze(-1)
    num_nodes_expected = torch.tensor(4) 
    coords_expected = torch.tensor([[0,0,0], [0,1,0], [1,0,0], [1,1,0]])
    force_expected = torch.tensor([[0], [1], [0], [0], [0], [1], [0], [0]])

    assert torch.equal(data.reciprocal_edge, reciprocal_edge_expected)
    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.directed_mask, directed_mask_expected)
    assert torch.equal(torch.tensor(data.num_nodes), num_nodes_expected)
    assert torch.equal(getattr(data, "coords"), coords_expected)
    assert torch.equal(getattr(data, "force"), force_expected)

