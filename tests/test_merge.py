import warnings

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


    data.merge_by_attribute("coords")

    # the two shared nodes are merged and come first; the nodes only one grid has keep their order
    coords_expected = torch.tensor([[1., 0., 0.], [1., 1., 0.], [0., 0., 0.], [0., 1., 0.], [2., 0., 0.], [2., 1., 0.]])

    num_nodes_expected = 6
    edge_index_expected = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], [1, 2, 4, 0, 3, 5, 0, 3, 1, 2, 0, 5, 1, 4]])
    directed_mask_expected = torch.tensor([True, False, True, False, False, True, True, True, True, False, False, True, False, False]).unsqueeze(-1)
    reciprocal_edge_expected = torch.tensor([3, 6, 10, 0, 8, 12, 1, 9, 4, 7, 2, 13, 5, 11]).unsqueeze(-1)

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

    # the merged nodes sit at x=1 and x=5, so only the survivors' coords are kept
    with pytest.warns(UserWarning, match=r"Merged nodes differ in \['coords'\]"):
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




def test_merge_by_attribute_leaves_nodes_with_unique_values_alone():

    data = StructData(node_attrs={"coords": torch.empty((0, 3), dtype=torch.float)}, edge_attrs={"force": torch.empty((0, 1), dtype=torch.float)})
    data.add_nodes(coords=torch.tensor([[3., 0., 0.], [1., 0., 0.], [2., 0., 0.]]))
    data.add_edges(edge_indices=torch.tensor([[0, 1], [2, 2]]), force=torch.tensor([[1.], [2.]]))
    edge_index_before = data.edge_index.clone()

    # no two nodes coincide, so nothing is renumbered or reordered
    data.merge_by_attribute("coords")

    assert data.coords[:, 0].tolist() == [3., 1., 2.]
    assert torch.equal(data.edge_index, edge_index_before)


def test_merge_by_attribute_rounds_on_shifted_grids():

    # the largest coordinate is 10, so the tolerance is 1e-5 * 10 = 1e-4
    data = StructData(node_attrs={"coords": torch.empty((0, 3), dtype=torch.float)})
    data.add_nodes(coords=torch.tensor([
        [0.00001, 0., 0.], [0.00002, 0., 0.],      # in the same cell of the unshifted grid
        [5.000048, 0., 0.], [5.000052, 0., 0.],    # right on either side of an edge of the unshifted grid, at 5.00005
        [9., 0., 0.], [9.0005, 0., 0.],            # further apart than the tolerance
        [10., 0., 0.],
    ]))

    data.merge_by_attribute("coords")

    # the first two pairs are merged, the third pair is not
    assert data.num_nodes == 5
    assert sorted(round(x, 4) for x in data.coords[:, 0].tolist()) == [0., 5., 9., 9.0005, 10.]


def test_merge_by_attribute_with_mask():

    data = StructData(node_attrs={"coords": torch.empty((0, 3), dtype=torch.float)})
    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [0., 0., 0.], [5., 0., 0.], [5., 0., 0.]]))

    # only the last two nodes may be merged
    data.merge_by_attribute("coords", mask=torch.tensor([False, False, True, True]))

    assert data.num_nodes == 3
    assert sorted(data.coords[:, 0].tolist()) == [0., 0., 5.]


def test_merge_by_attribute_bool_and_int():

    # only True nodes are merged
    data = StructData(node_attrs={"flag": torch.empty((0, 1), dtype=torch.bool)})
    data.add_nodes(flag=torch.tensor([[True], [False], [False], [True]]))
    data.merge_by_attribute("flag")
    assert sorted(data.flag.view(-1).tolist()) == [False, False, True]

    # int values are compared exactly
    data = StructData(node_attrs={"label": torch.empty((0, 1), dtype=torch.long)})
    data.add_nodes(label=torch.tensor([[1], [2], [1]]))
    data.merge_by_attribute("label")
    assert sorted(data.label.view(-1).tolist()) == [1, 2]


def test_merge_warns_when_merged_nodes_differ():

    def build():
        data = StructData(node_attrs={
            "coords": torch.empty((0, 3), dtype=torch.float),
            "load": torch.empty((0, 1), dtype=torch.float),
        })
        data.add_nodes(coords=torch.zeros(2, 3), load=torch.tensor([[1.], [2.]]))
        return data

    # without a priority, the survivor's load wins silently otherwise
    data = build()
    with pytest.warns(UserWarning, match=r"Merged nodes differ in \['load'\]"):
        data.merge(torch.tensor([0, 0]))
    assert data.load.tolist() == [[1.]]

    # with a priority, the survivor was chosen on purpose
    data = build()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        data.merge(torch.tensor([0, 0]), node_priority_attribute="load")
    assert data.load.tolist() == [[2.]]


def test_merge_warns_when_merged_edges_differ():

    def build():
        data = StructData(node_attrs={"coords": torch.empty((0, 3), dtype=torch.float)}, edge_attrs={"force": torch.empty((0, 1), dtype=torch.float)})
        data.add_nodes(coords=torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]))
        # merging nodes 0 and 1, which agree in every attribute, makes the edges 0-2 and 1-2 connect the same pair
        data.add_edges(edge_indices=torch.tensor([[0, 1], [2, 2]]), force=torch.tensor([[1.], [2.]]))
        return data

    data = build()
    with pytest.warns(UserWarning, match=r"Merged edges differ in \['force'\]"):
        data.merge(torch.tensor([0, 0, -1]))

    data = build()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        data.merge(torch.tensor([0, 0, -1]), edge_priority_attribute="force")


def test_merge_by_attribute_does_not_warn_about_its_own_attribute():

    data = StructData(node_attrs={"coords": torch.empty((0, 3), dtype=torch.float)})
    data.add_nodes(coords=torch.tensor([[0.000001, 0., 0.], [0.000002, 0., 0.]]))

    # the coords only agree within the tolerance, which merge compares them with as well; the unnamed nodes agree in
    # their name
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        data.merge_by_attribute("coords")

    assert data.num_nodes == 1


def test_merge_error_on_mismatch():

    def build():
        data = StructData(node_attrs={
            "coords": torch.empty((0, 3), dtype=torch.float),
            "load": torch.empty((0, 1), dtype=torch.float),
        })
        data.add_nodes(coords=torch.zeros(2, 3), load=torch.tensor([[1.], [2.]]))
        return data

    # instead of the warning, an error, raised before anything changes
    data = build()
    with pytest.raises(ValueError, match="Nodes 0 and 1 would be merged into one node, but their 'load' differs"):
        data.merge(torch.tensor([0, 0]), error_on_mismatch=True)
    assert data.num_nodes == 2

    # merge_by_attribute passes it on; the coords it groups by are not compared
    data = build()
    with pytest.raises(ValueError, match="their 'load' differs"):
        data.merge_by_attribute("coords", error_on_mismatch=True)
    assert data.num_nodes == 2

    # with a priority, the survivor was chosen on purpose, so there is nothing to raise
    data = build()
    data.merge(torch.tensor([0, 0]), node_priority_attribute="load", error_on_mismatch=True)
    assert data.load.tolist() == [[2.]]


def test_merge_error_on_mismatch_for_edges():

    data = StructData(node_attrs={"coords": torch.empty((0, 3), dtype=torch.float)}, edge_attrs={"force": torch.empty((0, 1), dtype=torch.float)})
    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]))
    # merging nodes 0 and 1 makes the edges 0-2 and 1-2 connect the same pair of nodes
    data.add_edges(edge_indices=torch.tensor([[0, 1], [2, 2]]), force=torch.tensor([[1.], [2.]]))
    edge_index_before = data.edge_index.clone()

    with pytest.raises(ValueError, match=r"Edges \(0, 2\) and \(1, 2\) would be merged into one edge, but their 'force' differs"):
        data.merge(torch.tensor([0, 0, -1]), error_on_mismatch=True)

    # raised before anything changed, the nodes included
    assert data.num_nodes == 3
    assert torch.equal(data.edge_index, edge_index_before)


def test_merge_checks_edge_priority_attribute_before_changing_anything():

    data = StructData(node_attrs={"coords": torch.empty((0, 3), dtype=torch.float)}, edge_attrs={"force": torch.empty((0, 1), dtype=torch.float)})
    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]))
    data.add_edges(edge_indices=torch.tensor([[0, 1], [2, 2]]), force=torch.tensor([[1.], [1.]]))

    with pytest.raises(ValueError, match="edge_priority_attribute 'stiffness' not found"):
        data.merge(torch.tensor([0, 0, -1]), edge_priority_attribute="stiffness")

    # the nodes were not merged before the error
    assert data.num_nodes == 3


def test_merge_by_attribute_passes_priorities_on():

    data = StructData(
        node_attrs={"coords": torch.empty((0, 3), dtype=torch.float), "load": torch.empty((0, 1), dtype=torch.float)},
        edge_attrs={"force": torch.empty((0, 1), dtype=torch.float)},
    )
    # nodes 0 and 1 coincide, but differ in load; the edges 0-2 and 1-2 then connect the same pair, but differ in force
    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]]), load=torch.tensor([[1.], [2.], [0.]]))
    data.add_edges(edge_indices=torch.tensor([[0, 1], [2, 2]]), force=torch.tensor([[1.], [2.]]))

    # the priorities choose the survivors, so there is nothing to warn about
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        data.merge_by_attribute("coords", node_priority_attribute="load", edge_priority_attribute="force")

    assert sorted(data.load.view(-1).tolist()) == [0., 2.]
    assert data.force.view(-1).tolist() == [2., 2.]
