import pytest
import torch
from torch_structure.data.data import StructData


def build_rot4_structure():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    # stiffness is not a copy attribute of the symmetry
    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
        "stiffness": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    data.add_symmetry({"rot4": data.create_rotational_symmetry(4)}, copy_attrs=["force"])

    # orbit 0 (nodes 0-3) and orbit 1 (nodes 4-7)
    data.add_nodes(symmetry="rot4", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))

    # radial edges: rows 0-3 (0 -> 4, 1 -> 5, 2 -> 6, 3 -> 7), ring edges: rows 4-7 (0 -> 1, 1 -> 2, 2 -> 3, 3 -> 0),
    # reciprocal rows 8-15
    data.add_edges(edge_indices=torch.tensor([[0, 0], [4, 1]]), force=torch.tensor([[3.], [-1.]]))

    return data


def test_set_edge_attr_propagates_to_edge_family():

    data = build_rot4_structure()

    # select the radial edge 2 -> 6 by its reciprocal row (row 10: 6 -> 2)
    mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
    mask[10] = True
    data.set_edge_attr("force", mask, torch.tensor([[7.]]))

    # with consider_symmetry=False only the ring edge 1 -> 2 (row 5) and its reciprocal row change
    mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
    mask[5] = True
    data.set_edge_attr("force", mask, torch.tensor([[4.]]), consider_symmetry=False)

    force_expected = torch.tensor(
        [[7.]] * 4 + [[-1.], [4.], [-1.], [-1.]] + [[7.]] * 4 + [[-1.], [4.], [-1.], [-1.]]
    )

    assert torch.equal(data.force, force_expected)


def test_set_edge_attr_with_combined_symmetry():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    mirror = data.create_mirror_symmetry(normal=torch.tensor([0., 1., 0.]))
    rotation = data.create_rotational_symmetry(4)
    data.add_symmetry({"d4": data.combine_symmetry(mirror, rotation)}, copy_attrs=["force"])

    # nodes 0-7: one orbit, node 8: outside any symmetry
    data.add_nodes(symmetry="d4", coords=torch.tensor([[2., 1., 0.]]))
    data.add_nodes(coords=torch.tensor([[0., 0., 3.]]))

    data.add_edges(edge_indices=torch.tensor([[0, 0, 8, 8], [1, 3, 0, 1]]), force=torch.zeros((4, 1)))

    # the edges to node 8 are added as given and come first (rows 0-1), then the mirror family
    # (rows 2-5) and the sector family (rows 6-9); rows 10-19 are the reciprocal rows
    edge_index_forward_expected = torch.tensor([[8, 8, 0, 2, 4, 6, 0, 2, 4, 6], [0, 1, 1, 3, 5, 7, 3, 5, 7, 1]])

    assert torch.equal(data.edge_index[:, :10], edge_index_forward_expected)

    # one edge of each family in the same call: row 4 (4 -> 5) and row 8 (4 -> 7)
    mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
    mask[[4, 8]] = True
    data.set_edge_attr("force", mask, torch.tensor([[2.], [9.]]))

    # an edge to node 8 is set, but not propagated
    mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
    mask[0] = True
    data.set_edge_attr("force", mask, torch.tensor([[5.]]))

    force_expected = torch.tensor(
        [[5.], [0.]] + [[2.]] * 4 + [[9.]] * 4 + [[5.], [0.]] + [[2.]] * 4 + [[9.]] * 4
    )

    assert torch.equal(data.force, force_expected)


def test_set_edge_attr_invalid_input():

    data = build_rot4_structure()

    num_rows = data.edge_index.shape[1]

    # the forward row 0 (0 -> 4) and its reciprocal row 8 (4 -> 0)
    mask = torch.zeros(num_rows, dtype=torch.bool)
    mask[[0, 8]] = True

    with pytest.raises(ValueError, match="both the forward and reciprocal row"):
        data.set_edge_attr("force", mask, torch.tensor([[1.], [2.]]))

    mask = torch.zeros(num_rows, dtype=torch.bool)
    mask[0] = True

    with pytest.raises(ValueError, match="one row per masked edge"):
        data.set_edge_attr("force", mask, torch.tensor([[1.], [2.]]))

    with pytest.raises(ValueError, match="not registered as a copy attribute"):
        data.set_edge_attr("stiffness", mask, torch.tensor([[1.]]))

    with pytest.raises(ValueError, match=f"mask must have {num_rows} entries"):
        data.set_edge_attr("force", torch.zeros(num_rows - 1, dtype=torch.bool), torch.zeros((0, 1)))

    # two radial edges, i.e. symmetry copies of each other, with different values
    mask = torch.zeros(num_rows, dtype=torch.bool)
    mask[[0, 2]] = True

    with pytest.raises(ValueError, match="symmetry copies of each other"):
        data.set_edge_attr("force", mask, torch.tensor([[1.], [2.]]))

    # with the same value for both there is no conflict
    data.set_edge_attr("force", mask, torch.tensor([[1.], [1.]]))

    assert torch.equal(data.force[:4], torch.ones((4, 1)))
