import pytest
import torch
from torch_structure.data.data import StructData
import matplotlib.pyplot as plt

    
def test_merge_one_grid_to_cylinder():

    print("test_merge_one_grid_to_cylinder is not yet implemented")


def test_merge_with_changed_node_priority():

    print("test_merge_with_changed_node_priority is not yet implemented")


def test_merge_with_changed_edge_priority():

    print("test_merge_with_changed_edge_priority is not yet implemented")


def test_merge_two_grids_attribute_based():
        
    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(
        node_attrs=node_attrs,
    )

    #prepare and add first grid
    def coords_first_grid(u_ind, v_ind):

        return torch.hstack([u_ind,v_ind,torch.zeros_like(u_ind)])


    node_attrs_first_grid = {
        "coords": coords_first_grid
    }

    data.add_grid(2,2, node_attrs=node_attrs_first_grid)


    #prepare and add second grid
    def coords_second_grid(u_ind, v_ind):

        return torch.hstack([u_ind+1,v_ind,torch.zeros_like(u_ind)])
    

    node_attrs_second_grid = {
        "coords": coords_second_grid
    }

    data.add_grid(2,2, node_attrs=node_attrs_second_grid)


    data.merge_based_on_attributes("coords")

    coords_expected = torch.tensor([[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [1., 1., 0.], [2., 0., 0.], [2., 1., 0.]])
    
    num_nodes_expected = 6
    edge_index_expected = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5], [1, 2, 0, 3, 0, 3, 4, 1, 2, 5, 2, 5, 3, 4]])
    directed_mask_expected = torch.tensor([True, True, False, True, False, True, True, False, False, True, False, True, False, False]).unsqueeze(-1)
    reciprocal_edge_expected = torch.tensor([2,4,0,7,1,8,10,3,5,12,6,13,9,11]).unsqueeze(-1)

    assert torch.equal(data.coords, coords_expected)
    assert data.num_nodes == num_nodes_expected
    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.directed_mask, directed_mask_expected)
    assert torch.equal(data.reciprocal_edge, reciprocal_edge_expected)


def test_merge_two_grids_id_based():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(
        node_attrs=node_attrs,
    )

    #prepare and add first grid
    def coords_first_grid(u_ind, v_ind):

        return torch.hstack([u_ind,v_ind,torch.zeros_like(u_ind)])


    node_attrs_first_grid = {
        "coords": coords_first_grid
    }

    data.add_grid(2,2, node_attrs=node_attrs_first_grid)


    #prepare and add second grid
    def coords_second_grid(u_ind, v_ind):

        return torch.hstack([u_ind+5,v_ind,torch.zeros_like(u_ind)])
    

    node_attrs_second_grid = {
        "coords": coords_second_grid
    }

    data.add_grid(2,2, node_attrs=node_attrs_second_grid)

    #create merge ids and merge
    merge_ids = torch.tensor([-1, -1, 1, 2, 1, 2, -1, -1])
    data.merge(merge_group_id = merge_ids)

    coords_expected = torch.tensor([[1,0,0], [1,1,0], [0,0,0], [0,1,0], [6,0,0], [6,1,0]])
    num_nodes_expected = 6
    edge_index_expected = torch.tensor([[0,0,0,1,1,1,2,2,3,3,4,4,5,5],[1,2,4,0,3,5,0,3,1,2,0,5,1,4]])
    directed_mask_expected = torch.tensor([True,False,True,False,False,True,True,True,True,False,False,True,False,False]).unsqueeze(-1)
    reciprocal_edge_expected = torch.tensor([3,6,10,0,8,12,1,9,4,7,2,13,5,11]).unsqueeze(-1)

    assert torch.equal(data.coords, coords_expected)
    assert data.num_nodes == num_nodes_expected
    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.directed_mask, directed_mask_expected)
    assert torch.equal(data.reciprocal_edge, reciprocal_edge_expected)


