import pytest
import torch
from torch_structure.data.data import StructData


def build_structure(with_pattern=True, nodes_by_pattern=True):

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    if with_pattern:
        data.add_patterns({"rot4": data.create_rotational_pattern(4)})

    if nodes_by_pattern:
        # expansion 0 (nodes 0-3) and expansion 1 (nodes 4-7)
        data.add_nodes_by_pattern(
            pattern="rot4",
            coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]),
            load=torch.tensor([[0., 0., -1.], [0., 0., -2.]]),
        )

    return data


def test_set_node_attr_per_expansion_by_pattern_transform_attribute():

    data = build_structure()

    # node 8 is outside any pattern and must not be mistaken for expansion 1, position 3
    data.add_nodes(coords=torch.tensor([[0., 0., 3.]]))

    data.set_node_attr_per_expansion_by_pattern(
        "coords", expansion_ids=[1], expansion_positions=[3], value=torch.tensor([[0., -3., 1.]])
    )

    # expansion 1 is rebuilt from position 3; expansion 0 and node 8 are unchanged
    coords_expected = torch.tensor([
        [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.],
        [3., 0., 1.], [0., 3., 1.], [-3., 0., 1.], [0., -3., 1.],
        [0., 0., 3.],
    ])

    assert torch.allclose(data.coords, coords_expected, atol=1e-6)


def test_set_node_attr_per_expansion_several_expansions_and_single_node():

    data = build_structure()

    # two expansions in one call, given out of order and addressed via different positions
    data.set_node_attr_per_expansion_by_pattern(
        "load", expansion_ids=[1, 0], expansion_positions=[2, 1], value=torch.tensor([[0., 0., -6.], [0., 0., -3.]])
    )

    # set_node_attr_per_expansion only changes expansion 0, position 2 (node 2)
    data.set_node_attr_per_expansion(
        "load", expansion_ids=[0], expansion_positions=[2], value=torch.tensor([[1., 0., 0.]])
    )

    load_expected = torch.tensor([
        [0., 0., -3.], [0., 0., -3.], [1., 0., 0.], [0., 0., -3.],
        [0., 0., -6.], [0., 0., -6.], [0., 0., -6.], [0., 0., -6.],
    ])

    assert torch.equal(data.load, load_expected)


def test_set_node_attr_per_expansion_by_pattern_with_replicate_levels():

    data = build_structure()

    # rot4 has a single level: frozen, only expansion 1, position 2 (node 6) changes
    data.set_node_attr_per_expansion_by_pattern(
        "load", expansion_ids=[1], expansion_positions=[2], value=torch.tensor([[1., 1., 1.]]), replicate_levels=[False]
    )

    load_expected = torch.tensor([[0., 0., -1.]] * 4 + [[0., 0., -2.]] * 2 + [[1., 1., 1.]] + [[0., 0., -2.]])

    assert torch.equal(data.load, load_expected)


def test_set_node_attr_per_expansion_same_expansion():

    data = build_structure()

    # two positions of the same expansion, even with the same value
    with pytest.raises(ValueError, match="same expansion"):
        data.set_node_attr_per_expansion_by_pattern("load", [0, 0], [1, 2], torch.zeros((2, 3)))

    # set_node_attr_per_expansion sets both
    data.set_node_attr_per_expansion("load", [0, 0], [1, 2], torch.zeros((2, 3)))

    load_expected = torch.tensor([[0., 0., -1.], [0., 0., 0.], [0., 0., 0.], [0., 0., -1.]] + [[0., 0., -2.]] * 4)

    assert torch.equal(data.load, load_expected)


@pytest.mark.parametrize("method", ["set_node_attr_per_expansion", "set_node_attr_per_expansion_by_pattern"])
def test_set_node_attr_per_expansion_invalid_input(method):

    data = build_structure()

    # expansion 2 does not exist
    with pytest.raises(ValueError, match="No node found"):
        getattr(data, method)("load", [2], [0], torch.zeros((1, 3)))

    # a pattern is registered, but no node belongs to an expansion
    data = build_structure(nodes_by_pattern=False)
    data.add_nodes(coords=torch.tensor([[0., 0., 0.]]))

    with pytest.raises(ValueError, match="No node found"):
        getattr(data, method)("load", [0], [0], torch.zeros((1, 3)))

    # no pattern registered
    data = build_structure(with_pattern=False, nodes_by_pattern=False)
    data.add_nodes(coords=torch.tensor([[0., 0., 0.]]))

    with pytest.raises(AttributeError):
        getattr(data, method)("load", [0], [0], torch.zeros((1, 3)))


def test_set_node_attr_per_expansion_by_pattern_refuses_a_merged_position_on_a_frozen_level():

    data = StructData(node_attrs={
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    })
    x = torch.tensor([1., 0., 0.])
    data.add_patterns({"s": data.nest_patterns([data.create_translational_pattern(4, x), data.create_mirror_pattern(normal=x)])})

    # the array starts on the mirror plane, so position 4, its mirror image, is merged into position 0
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[0., 0., 0.]]), load=torch.zeros(1, 3))

    # with the mirror frozen, position 4 means the mirrored side, but position 0 the original one
    with pytest.raises(ValueError, match="differs on a level that replicate_levels freezes"):
        data.set_node_attr_per_expansion_by_pattern("load", [0], [4], torch.tensor([[0., 0., -5.]]), replicate_levels=[True, False])

    assert torch.equal(data.load, torch.zeros_like(data.load))

    # with every level replicated, both positions reach the whole expansion
    data.set_node_attr_per_expansion_by_pattern("load", [0], [4], torch.tensor([[0., 0., -5.]]))

    assert torch.equal(data.load, torch.tensor([[0., 0., -5.]]).expand(data.num_nodes, 3))


@pytest.mark.parametrize("method", ["set_node_attr_per_expansion", "set_node_attr_per_expansion_by_pattern"])
@pytest.mark.parametrize("num_rows", [0, 2])
def test_set_node_attr_per_expansion_needs_one_value_row_per_entry(method, num_rows):

    data = build_structure()
    load_before = data.load.clone()

    # one entry, but no or two rows of value
    with pytest.raises(ValueError, match=rf"value must have one row per entry \(1\), but has {num_rows}"):
        getattr(data, method)("load", [0], [0], torch.zeros((num_rows, 3)))

    assert torch.equal(data.load, load_before)
