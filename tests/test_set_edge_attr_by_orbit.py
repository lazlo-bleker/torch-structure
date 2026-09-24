import pytest
import torch
from torch_structure.data.data import StructData


def build_structure(symmetric_nodes=True):

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    data.add_symmetry({"rot4": data.create_rotational_symmetry(4)}, copy_attrs=["force"])

    if symmetric_nodes:
        # orbit 0 (nodes 0-3) and orbit 1 (nodes 4-7)
        data.add_nodes(symmetry="rot4", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))

        # radial edges: rows 0-3 (0 -> 4, 1 -> 5, 2 -> 6, 3 -> 7), ring edges: rows 4-7 (0 -> 1, 1 -> 2, 2 -> 3,
        # 3 -> 0), reciprocal rows 8-15
        data.add_edges(edge_indices=torch.tensor([[0, 0], [4, 1]]), force=torch.tensor([[3.], [-1.]]))

    return data


def test_set_edge_attr_by_orbit_propagates():

    data = build_structure()

    # node 8 is outside any symmetry and must not be mistaken for orbit 1, position 3
    data.add_nodes(coords=torch.tensor([[0., 0., 3.]]))

    # the radial edge from orbit 0, position 3 to orbit 1, position 3
    data.set_edge_attr_by_orbit(
        "force", src_orbit_ids=[0], dest_orbit_ids=[1], src_orbit_positions=[3], dest_orbit_positions=[3],
        value=torch.tensor([[6.]]),
    )

    # all radial edges change, the ring edges do not
    force_expected = torch.tensor([[6.]] * 4 + [[-1.]] * 4 + [[6.]] * 4 + [[-1.]] * 4)

    assert torch.equal(data.force, force_expected)


def test_set_edge_attr_by_orbit_reversed_direction_and_without_symmetry():

    data = build_structure()

    # orbit 1, position 1 -> orbit 0, position 1 is the reciprocal row of the radial edge 1 -> 5
    data.set_edge_attr_by_orbit("force", [1], [0], [1], [1], torch.tensor([[7.]]))

    # with consider_symmetry=False only the ring edge 0 -> 1 (row 4) and its reciprocal row change
    data.set_edge_attr_by_orbit("force", [0], [0], [0], [1], torch.tensor([[4.]]), consider_symmetry=False)

    force_expected = torch.tensor(
        [[7.]] * 4 + [[4.], [-1.], [-1.], [-1.]] + [[7.]] * 4 + [[4.], [-1.], [-1.], [-1.]]
    )

    assert torch.equal(data.force, force_expected)


def test_set_edge_attr_by_orbit_invalid_input():

    data = build_structure()

    # orbit 0, position 0 (node 0) and orbit 1, position 2 (node 6) are not connected
    with pytest.raises(ValueError, match="No edge found"):
        data.set_edge_attr_by_orbit("force", [0], [1], [0], [2], torch.tensor([[1.]]))

    # orbit 3 does not exist
    with pytest.raises(ValueError, match="No node found"):
        data.set_edge_attr_by_orbit("force", [3], [1], [0], [0], torch.tensor([[1.]]))

    # a symmetry is registered, but no node belongs to an orbit
    data = build_structure(symmetric_nodes=False)
    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [1., 0., 0.]]))
    data.add_edges(edge_indices=torch.tensor([[0], [1]]), force=torch.tensor([[1.]]))

    with pytest.raises(ValueError, match="No node found"):
        data.set_edge_attr_by_orbit("force", [0], [0], [0], [1], torch.tensor([[1.]]))
