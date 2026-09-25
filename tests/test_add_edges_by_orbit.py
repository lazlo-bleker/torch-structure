import pytest
import torch
from torch_structure.data.data import StructData


def build_structure(with_symmetry=True, symmetric_nodes=True):

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    if with_symmetry:
        data.add_symmetry({"rot4": data.create_rotational_symmetry(4)})

    if symmetric_nodes:
        # orbit 0 (nodes 0-3) and orbit 1 (nodes 4-7)
        data.add_nodes(symmetry="rot4", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))

    return data


def test_add_edges_by_orbit_all_copies():

    data = build_structure()

    # node 8 is outside any symmetry and must not be mistaken for orbit 1, position 3
    data.add_nodes(coords=torch.tensor([[0., 0., 3.]]))

    data.add_edges_by_orbit(
        src_orbit_ids=[0], dest_orbit_ids=[1], src_orbit_positions=[3], dest_orbit_positions=[3],
        force=torch.tensor([[2.]]),
    )

    # all 4 copies are added, whichever orbit position was given
    edge_index_expected = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 0, 1, 2, 3]])
    force_expected = torch.full((8, 1), 2.)

    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.force, force_expected)


def test_add_edges_by_orbit_without_symmetry():

    data = build_structure()

    # orbit 0, position 0 -> orbit 0, position 1 and orbit 0, position 2 -> orbit 1, position 3
    data.add_edges_by_orbit(
        src_orbit_ids=[0, 0], dest_orbit_ids=[0, 1], src_orbit_positions=[0, 2], dest_orbit_positions=[1, 3],
        consider_symmetry=False, force=torch.tensor([[1.], [2.]]),
    )

    edge_index_expected = torch.tensor([[0, 2, 1, 7], [1, 7, 0, 2]])
    force_expected = torch.tensor([[1.], [2.], [1.], [2.]])

    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.force, force_expected)


def test_add_edges_by_orbit_invalid_input():

    data = build_structure()

    # orbit 5 does not exist
    with pytest.raises(ValueError, match="No node found"):
        data.add_edges_by_orbit([5], [0], [0], [0])

    # position 4 does not exist in a 4-fold symmetry
    with pytest.raises(ValueError, match="No node found"):
        data.add_edges_by_orbit([0], [1], [0], [4])

    # a symmetry is registered, but no node belongs to an orbit
    data = build_structure(symmetric_nodes=False)
    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [1., 0., 0.]]))

    with pytest.raises(ValueError, match="No node found"):
        data.add_edges_by_orbit([0], [0], [0], [1])

    # no symmetry registered
    data = build_structure(with_symmetry=False, symmetric_nodes=False)
    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [1., 0., 0.]]))

    with pytest.raises(AttributeError):
        data.add_edges_by_orbit([0], [0], [0], [1])
