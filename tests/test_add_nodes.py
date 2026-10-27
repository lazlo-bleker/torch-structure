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


def test_add_nodes_with_rotational_pattern():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    data.add_patterns({"rot4": data.create_rotational_pattern(4)})

    new_node_attrs = {
        "coords": torch.tensor([[1., 0., 0.], [2., 0., 0.]]),
        "load": torch.tensor([[0., 0., -1.], [0., 0., -2.]]),
    }

    data.add_nodes_by_pattern(names=["A", "B"], pattern="rot4", **new_node_attrs)

    # each seed becomes an expansion of 4 nodes, stored one after the other and ordered by expansion position;
    # coords are rotated per expansion position, load is copied
    num_nodes_expected = 8
    coords_expected = torch.tensor([
        [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.],
        [2., 0., 0.], [0., 2., 0.], [-2., 0., 0.], [0., -2., 0.],
    ])
    load_expected = torch.tensor([[0., 0., -1.]] * 4 + [[0., 0., -2.]] * 4)
    expansion_id_expected = torch.tensor([[0]] * 4 + [[1]] * 4)
    expansion_position_expected = torch.tensor([[0], [1], [2], [3]] * 2)
    pattern_id_expected = torch.zeros((8, 1), dtype=torch.long)
    nodes_named_a_expected = torch.tensor([0, 1, 2, 3])

    assert data.num_nodes == num_nodes_expected
    assert torch.allclose(data.coords, coords_expected, atol=1e-6)
    assert torch.equal(data.load, load_expected)
    assert torch.equal(data.expansion_id, expansion_id_expected)
    assert torch.equal(data.expansion_position, expansion_position_expected)
    assert torch.equal(data.pattern_id, pattern_id_expected)
    # the name is copied to every node of the expansion
    assert torch.equal(data.get_node_index_from_name("A"), nodes_named_a_expected)


def test_add_nodes_with_combined_pattern():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    mirror = data.create_mirror_pattern(normal=torch.tensor([0., 1., 0.]))
    rotation = data.create_rotational_pattern(4)
    data.add_patterns({"d4": data.nest_patterns([mirror, rotation])})

    data.add_nodes_by_pattern(pattern="d4", coords=torch.tensor([[2., 1., 0.], [3., 1., 5.]]))

    # expansion position p is the seed, mirrored if p is odd, then rotated by 90° * (p // 2)
    num_nodes_expected = 16
    coords_expected = torch.tensor([
        [2., 1., 0.], [2., -1., 0.], [-1., 2., 0.], [1., 2., 0.],
        [-2., -1., 0.], [-2., 1., 0.], [1., -2., 0.], [-1., -2., 0.],
        [3., 1., 5.], [3., -1., 5.], [-1., 3., 5.], [1., 3., 5.],
        [-3., -1., 5.], [-3., 1., 5.], [1., -3., 5.], [-1., -3., 5.],
    ])
    expansion_id_expected = torch.tensor([[0]] * 8 + [[1]] * 8)
    expansion_position_expected = torch.arange(8).repeat(2).unsqueeze(1)

    assert data.num_nodes == num_nodes_expected
    assert torch.allclose(data.coords, coords_expected, atol=1e-6)
    assert torch.equal(data.expansion_id, expansion_id_expected)
    assert torch.equal(data.expansion_position, expansion_position_expected)


def test_add_nodes_with_and_without_pattern():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    data.add_patterns({"mirror": data.create_mirror_pattern()})

    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [0., 0., 1.]]))
    data.add_nodes_by_pattern(pattern="mirror", coords=torch.tensor([[1., 0., 0.]]))
    data.add_nodes(coords=torch.tensor([[0., 0., 2.]]))
    data.add_nodes_by_pattern(pattern="mirror", coords=torch.tensor([[2., 0., 0.]]))

    # nodes added without a pattern get -1; expansion ids continue across the additions
    num_nodes_expected = 7
    coords_expected = torch.tensor([
        [0., 0., 0.], [0., 0., 1.], [1., 0., 0.], [-1., 0., 0.], [0., 0., 2.], [2., 0., 0.], [-2., 0., 0.],
    ])
    expansion_id_expected = torch.tensor([[-1], [-1], [0], [0], [-1], [1], [1]])
    expansion_position_expected = torch.tensor([[-1], [-1], [0], [1], [-1], [0], [1]])
    pattern_id_expected = torch.tensor([[-1], [-1], [0], [0], [-1], [0], [0]])

    assert data.num_nodes == num_nodes_expected
    assert torch.equal(data.coords, coords_expected)
    assert torch.equal(data.expansion_id, expansion_id_expected)
    assert torch.equal(data.expansion_position, expansion_position_expected)
    assert torch.equal(data.pattern_id, pattern_id_expected)

    with pytest.raises(ValueError, match="not registered"):
        data.add_nodes_by_pattern(pattern="rot4", coords=torch.tensor([[1., 0., 0.]]))


def test_add_nodes_by_pattern_without_pattern():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    with pytest.raises(ValueError, match="not registered"):
        data.add_nodes_by_pattern(pattern="rot4", coords=torch.tensor([[1., 0., 0.]]))

    assert data.num_nodes == 0


def test_add_nodes_rejects_pattern_bookkeeping():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    # without a registered pattern the bookkeeping attributes do not exist
    with pytest.raises(ValueError, match="Unexpected node attributes"):
        data.add_nodes(coords=torch.tensor([[0., 0., 5.]]), expansion_id=torch.tensor([[0]]))

    data.add_patterns({"rot4": data.create_rotational_pattern(4)})

    # expansion 0 (nodes 0-3)
    data.add_nodes_by_pattern(pattern="rot4", coords=torch.tensor([[1., 0., 0.]]))

    # a node added without a pattern cannot claim a position in expansion 0
    with pytest.raises(ValueError, match="pattern bookkeeping"):
        data.add_nodes(
            coords=torch.tensor([[0., 0., 5.]]),
            expansion_id=torch.tensor([[0]]),
            expansion_position=torch.tensor([[0]]),
            pattern_id=torch.tensor([[0]]),
        )

    # add_nodes_by_pattern sets the bookkeeping itself
    with pytest.raises(ValueError, match="pattern bookkeeping"):
        data.add_nodes_by_pattern(pattern="rot4", coords=torch.tensor([[2., 0., 0.]]), expansion_id=torch.tensor([[5]]))

    # the failed calls left the graph unchanged
    assert data.num_nodes == 4
    assert data.coords.shape == (4, 3)
    assert torch.equal(data.expansion_id, torch.zeros((4, 1), dtype=torch.long))
    assert torch.equal(data.expansion_position, torch.arange(4).unsqueeze(1))
