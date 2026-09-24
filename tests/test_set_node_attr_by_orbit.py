import pytest
import torch
from torch_structure.data.data import StructData


def build_structure(with_symmetry=True, symmetric_nodes=True):

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    if with_symmetry:
        data.add_symmetry({"rot4": data.create_rotational_symmetry(4)}, copy_attrs=["load"])

    if symmetric_nodes:
        # orbit 0 (nodes 0-3) and orbit 1 (nodes 4-7)
        data.add_nodes(
            symmetry="rot4",
            coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]),
            load=torch.tensor([[0., 0., -1.], [0., 0., -2.]]),
        )

    return data


def test_set_node_attr_by_orbit_transform_attribute():

    data = build_structure()

    # node 8 is outside any symmetry and must not be mistaken for orbit 1, position 3
    data.add_nodes(coords=torch.tensor([[0., 0., 3.]]))

    data.set_node_attr_by_orbit("coords", orbit_ids=[1], orbit_positions=[3], value=torch.tensor([[0., -3., 1.]]))

    # orbit 1 is rebuilt from position 3; orbit 0 and node 8 are unchanged
    coords_expected = torch.tensor([
        [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.],
        [3., 0., 1.], [0., 3., 1.], [-3., 0., 1.], [0., -3., 1.],
        [0., 0., 3.],
    ])

    assert torch.allclose(data.coords, coords_expected, atol=1e-6)


def test_set_node_attr_by_orbit_several_orbits_and_single_node():

    data = build_structure()

    # two orbits in one call, given out of order and addressed via different positions
    data.set_node_attr_by_orbit(
        "load", orbit_ids=[1, 0], orbit_positions=[2, 1], value=torch.tensor([[0., 0., -6.], [0., 0., -3.]])
    )

    # with consider_symmetry=False only orbit 0, position 2 (node 2) changes
    data.set_node_attr_by_orbit(
        "load", orbit_ids=[0], orbit_positions=[2], value=torch.tensor([[1., 0., 0.]]), consider_symmetry=False
    )

    load_expected = torch.tensor([
        [0., 0., -3.], [0., 0., -3.], [1., 0., 0.], [0., 0., -3.],
        [0., 0., -6.], [0., 0., -6.], [0., 0., -6.], [0., 0., -6.],
    ])

    assert torch.equal(data.load, load_expected)


def test_set_node_attr_by_orbit_invalid_input():

    data = build_structure()

    # orbit 2 does not exist
    with pytest.raises(ValueError, match="No node found"):
        data.set_node_attr_by_orbit("load", [2], [0], torch.zeros((1, 3)))

    # two positions of the same orbit, even with the same value
    with pytest.raises(ValueError, match="same orbit"):
        data.set_node_attr_by_orbit("load", [0, 0], [1, 2], torch.zeros((2, 3)))

    # a symmetry is registered, but no node belongs to an orbit
    data = build_structure(symmetric_nodes=False)
    data.add_nodes(coords=torch.tensor([[0., 0., 0.]]))

    with pytest.raises(ValueError, match="No node found"):
        data.set_node_attr_by_orbit("load", [0], [0], torch.zeros((1, 3)))

    # no symmetry registered
    data = build_structure(with_symmetry=False, symmetric_nodes=False)
    data.add_nodes(coords=torch.tensor([[0., 0., 0.]]))

    with pytest.raises(AttributeError):
        data.set_node_attr_by_orbit("load", [0], [0], torch.zeros((1, 3)))
