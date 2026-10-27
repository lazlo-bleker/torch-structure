import pytest
import torch
from torch_structure.data.data import StructData


def build_rotated_array():
    """An array of 5 along z (level 0) nested inside a 12-fold rotation (level 1), grown from two seeds (expansions 0 and
    1)."""
    data = StructData(node_attrs={
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    })
    rotated_array = data.nest_patterns([
        data.create_translational_pattern(5, direction=torch.tensor([0., 0., 1.])),
        data.create_rotational_pattern(12),
    ])
    data.add_patterns({"rotated_array": rotated_array})
    data.add_nodes_by_pattern(
        pattern="rotated_array", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]), load=torch.zeros(2, 3),
    )
    return data


def build_mirrored_array():
    """An array of 4 along x (level 0), mirrored at the plane x=0 (level 1), which it starts on: level positions
    [0, 1], the mirror image of the first node, are merged into [0, 0]."""
    data = StructData(node_attrs={
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    })
    x = torch.tensor([1., 0., 0.])
    mirrored_array = data.nest_patterns([data.create_translational_pattern(4, x), data.create_mirror_pattern(normal=x)])
    data.add_patterns({"mirrored_array": mirrored_array})
    data.add_nodes_by_pattern(pattern="mirrored_array", coords=torch.tensor([[0., 0., 0.]]), load=torch.zeros(1, 3))
    return data


@pytest.mark.parametrize("attr, value", [
    ("coords", torch.tensor([[0., 1.5, 2.]])),      # a transform attribute
    ("load", torch.tensor([[0., 0., -3.]])),        # a copied attribute
])
@pytest.mark.parametrize("replicate_levels", [None, [True, False], [False, True], [False, False]])
def test_set_node_attr_by_pattern_level_matches_set_node_attr_per_expansion_by_pattern(attr, value, replicate_levels):

    # level positions [2, 3]: the third node of the array in the fourth rotated copy, expansion position 2 + 5 * 3
    by_level = build_rotated_array()
    by_level.set_node_attr_by_pattern_level(attr, [0], [[2, 3]], value, replicate_levels=replicate_levels)

    per_expansion = build_rotated_array()
    per_expansion.set_node_attr_per_expansion_by_pattern(attr, [0], [17], value, replicate_levels=replicate_levels)

    assert torch.equal(getattr(by_level, attr), getattr(per_expansion, attr))


def test_set_node_attr_by_pattern_level_several_rows():

    value = torch.tensor([[0., 0., -1.], [0., 0., -2.]])

    by_level = build_rotated_array()
    by_level.set_node_attr_by_pattern_level("load", [0, 1], torch.tensor([[0, 0], [4, 11]]), value)

    per_expansion = build_rotated_array()
    per_expansion.set_node_attr_per_expansion_by_pattern("load", [0, 1], [0, 4 + 5 * 11], value)

    assert torch.equal(by_level.load, per_expansion.load)


def test_set_node_attr_by_pattern_level_at_a_merged_copy():

    data = build_mirrored_array()

    # [0, 1] was merged into [0, 0]; with every level replicated, it reaches the whole expansion
    data.set_node_attr_by_pattern_level("load", [0], [[0, 1]], torch.tensor([[0., 0., -5.]]))
    assert torch.equal(data.load, torch.tensor([[0., 0., -5.]]).expand(data.num_nodes, 3))

    # with the mirror frozen, [0, 1] means the mirrored side, but its node sits at [0, 0], on the original side
    with pytest.raises(ValueError, match="differs on a level that replicate_levels freezes"):
        data.set_node_attr_by_pattern_level(
            "load", [0], [[0, 1]], torch.tensor([[0., 0., -1.]]), replicate_levels=[True, False],
        )

    # the failed call left the attribute unchanged
    assert torch.equal(data.load, torch.tensor([[0., 0., -5.]]).expand(data.num_nodes, 3))


@pytest.mark.parametrize("expansion_ids, level_positions, value, message", [
    ([0], [[0]], torch.zeros(1, 3), r"level_positions must have shape \[1, 2\]"),
    ([0], [0, 0], torch.zeros(1, 3), r"level_positions must have shape \[1, 2\]"),
    ([0, 0], [[0, 0], [1]], torch.zeros(2, 3), "all of the same length"),
    ([0], [[0., 0.]], torch.zeros(1, 3), "must hold integers"),
    ([0], [[5, 0]], torch.zeros(1, 3), r"level 0 has 5 positions \(0 to 4\), got 5"),
    ([0], [[0, -1]], torch.zeros(1, 3), r"level 1 has 12 positions \(0 to 11\), got -1"),
    ([7], [[0, 0]], torch.zeros(1, 3), r"No expansion found with id\(s\) \[7\]"),
    ([0], [[0, 0]], torch.zeros(2, 3), r"value must have one row per entry \(1\), but has 2"),
])
def test_set_node_attr_by_pattern_level_rejects_invalid_input(expansion_ids, level_positions, value, message):

    data = build_rotated_array()

    with pytest.raises(ValueError, match=message):
        data.set_node_attr_by_pattern_level("load", expansion_ids, level_positions, value)

    # the failed call left the attribute unchanged
    assert torch.equal(data.load, torch.zeros(data.num_nodes, 3))


def test_set_node_attr_by_pattern_level_rejects_expansions_of_several_patterns():

    data = StructData(node_attrs={
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    })
    data.add_patterns({"rot4": data.create_rotational_pattern(4), "mirror": data.create_mirror_pattern()})

    # expansion 0: the rotation, expansion 1: the mirror
    data.add_nodes_by_pattern(pattern="rot4", coords=torch.tensor([[1., 0., 0.]]))
    data.add_nodes_by_pattern(pattern="mirror", coords=torch.tensor([[3., 1., 0.]]))

    with pytest.raises(ValueError, match="several registered patterns"):
        data.set_node_attr_by_pattern_level("load", [0, 1], [[0], [0]], torch.zeros(2, 3))


def test_set_node_attr_by_pattern_level_without_pattern():

    data = StructData(node_attrs={"load": torch.empty((0, 3), dtype=torch.float)})
    data.add_nodes(load=torch.zeros(2, 3))

    with pytest.raises(ValueError, match="No pattern is registered"):
        data.set_node_attr_by_pattern_level("load", [0], [[0]], torch.ones(1, 3))


def test_set_node_attr_by_pattern_level_without_rows():

    data = build_rotated_array()
    data.set_node_attr_by_pattern_level("load", [], [], torch.empty((0, 3)))

    assert torch.equal(data.load, torch.zeros(data.num_nodes, 3))
