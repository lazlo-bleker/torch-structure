import pytest
import torch
from torch_structure.data.data import StructData

def test_add_nodes():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    coords = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])

    new_node_attrs = {
        "coords": coords
    }

    data.add_nodes(**new_node_attrs)

    num_nodes_expected = torch.tensor(3)
    coords_expected = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])

    assert torch.equal(torch.tensor(data.num_nodes), num_nodes_expected)
    assert torch.equal(data.coords, coords_expected)


def test_add_nodes_with_rotational_symmetry():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    data.add_symmetry({"rot4": data.create_rotational_symmetry(4)}, copy_attrs=["load"])

    new_node_attrs = {
        "coords": torch.tensor([[1., 0., 0.], [2., 0., 0.]]),
        "load": torch.tensor([[0., 0., -1.], [0., 0., -2.]]),
    }

    data.add_nodes(names=["A", "B"], symmetry="rot4", **new_node_attrs)

    # each seed becomes an orbit of 4 nodes, stored one after the other and ordered by orbit position;
    # coords are rotated per orbit position, load is copied
    num_nodes_expected = 8
    coords_expected = torch.tensor([
        [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.],
        [2., 0., 0.], [0., 2., 0.], [-2., 0., 0.], [0., -2., 0.],
    ])
    load_expected = torch.tensor([[0., 0., -1.]] * 4 + [[0., 0., -2.]] * 4)
    orbit_id_expected = torch.tensor([[0]] * 4 + [[1]] * 4)
    orbit_position_expected = torch.tensor([[0], [1], [2], [3]] * 2)
    symmetry_id_expected = torch.zeros((8, 1), dtype=torch.long)
    nodes_named_a_expected = torch.tensor([0, 1, 2, 3])

    assert data.num_nodes == num_nodes_expected
    assert torch.allclose(data.coords, coords_expected, atol=1e-6)
    assert torch.equal(data.load, load_expected)
    assert torch.equal(data.orbit_id, orbit_id_expected)
    assert torch.equal(data.orbit_position, orbit_position_expected)
    assert torch.equal(data.symmetry_id, symmetry_id_expected)
    # the name is copied to every node of the orbit
    assert torch.equal(data.get_node_index_from_name("A"), nodes_named_a_expected)


def test_add_nodes_with_combined_symmetry():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    mirror = data.create_mirror_symmetry(normal=torch.tensor([0., 1., 0.]))
    rotation = data.create_rotational_symmetry(4)
    data.add_symmetry({"d4": data.combine_symmetry(mirror, rotation)})

    data.add_nodes(symmetry="d4", coords=torch.tensor([[2., 1., 0.], [3., 1., 5.]]))

    # orbit position p is the seed, mirrored if p is odd, then rotated by 90° * (p // 2)
    num_nodes_expected = 16
    coords_expected = torch.tensor([
        [2., 1., 0.], [2., -1., 0.], [-1., 2., 0.], [1., 2., 0.],
        [-2., -1., 0.], [-2., 1., 0.], [1., -2., 0.], [-1., -2., 0.],
        [3., 1., 5.], [3., -1., 5.], [-1., 3., 5.], [1., 3., 5.],
        [-3., -1., 5.], [-3., 1., 5.], [1., -3., 5.], [-1., -3., 5.],
    ])
    orbit_id_expected = torch.tensor([[0]] * 8 + [[1]] * 8)
    orbit_position_expected = torch.arange(8).repeat(2).unsqueeze(1)

    assert data.num_nodes == num_nodes_expected
    assert torch.allclose(data.coords, coords_expected, atol=1e-6)
    assert torch.equal(data.orbit_id, orbit_id_expected)
    assert torch.equal(data.orbit_position, orbit_position_expected)


def test_add_nodes_with_and_without_symmetry():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    data.add_symmetry({"mirror": data.create_mirror_symmetry()})

    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [0., 0., 1.]]))
    data.add_nodes(symmetry="mirror", coords=torch.tensor([[1., 0., 0.]]))
    data.add_nodes(coords=torch.tensor([[0., 0., 2.]]))
    data.add_nodes(symmetry="mirror", coords=torch.tensor([[2., 0., 0.]]))

    # nodes added without a symmetry get -1; orbit ids continue across the additions
    num_nodes_expected = 7
    coords_expected = torch.tensor([
        [0., 0., 0.], [0., 0., 1.], [1., 0., 0.], [-1., 0., 0.], [0., 0., 2.], [2., 0., 0.], [-2., 0., 0.],
    ])
    orbit_id_expected = torch.tensor([[-1], [-1], [0], [0], [-1], [1], [1]])
    orbit_position_expected = torch.tensor([[-1], [-1], [0], [1], [-1], [0], [1]])
    symmetry_id_expected = torch.tensor([[-1], [-1], [0], [0], [-1], [0], [0]])

    assert data.num_nodes == num_nodes_expected
    assert torch.equal(data.coords, coords_expected)
    assert torch.equal(data.orbit_id, orbit_id_expected)
    assert torch.equal(data.orbit_position, orbit_position_expected)
    assert torch.equal(data.symmetry_id, symmetry_id_expected)

    with pytest.raises(ValueError, match="not registered"):
        data.add_nodes(symmetry="rot4", coords=torch.tensor([[1., 0., 0.]]))



