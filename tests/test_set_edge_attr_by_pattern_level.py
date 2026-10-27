import pytest
import torch
from torch_structure.data.data import StructData


def build_rotated_array():
    """An array of 5 along z (level 0) nested inside a 12-fold rotation (level 1), grown from one seed (expansion 0), with
    the steps along the array (force 1) and around the rotation (force 2) as edges."""
    data = StructData(
        node_attrs={"coords": torch.empty((0, 3), dtype=torch.float)},
        edge_attrs={"force": torch.empty((0, 1), dtype=torch.float)},
    )
    rotated_array = data.nest_patterns([
        data.create_translational_pattern(5, direction=torch.tensor([0., 0., 1.])),
        data.create_rotational_pattern(12),
    ])
    data.add_patterns({"rotated_array": rotated_array})
    data.add_nodes_by_pattern(pattern="rotated_array", coords=torch.tensor([[1., 0., 0.]]))
    data.add_edges_by_pattern_level(
        src_expansion_ids=[0, 0], dest_expansion_ids=[0, 0],
        src_level_positions=[[0, 0], [0, 0]], dest_level_positions=[[1, 0], [0, 1]],
        force=torch.tensor([[1.], [2.]]),
    )
    return data


def build_mirrored_array():
    """An array of 4 along x (level 0), mirrored at the plane x=0 (level 1), which it starts on, with the steps along
    the array as edges. Level positions [0, 1], the mirror image of the first node, are merged into [0, 0]."""
    data = StructData(
        node_attrs={"coords": torch.empty((0, 3), dtype=torch.float)},
        edge_attrs={"force": torch.empty((0, 1), dtype=torch.float)},
    )
    x = torch.tensor([1., 0., 0.])
    mirrored_array = data.nest_patterns([data.create_translational_pattern(4, x), data.create_mirror_pattern(normal=x)])
    data.add_patterns({"mirrored_array": mirrored_array})
    data.add_nodes_by_pattern(pattern="mirrored_array", coords=torch.tensor([[0., 0., 0.]]))
    data.add_edges_by_pattern_level(
        src_expansion_ids=[0], dest_expansion_ids=[0], src_level_positions=[[0, 0]], dest_level_positions=[[1, 0]],
        force=torch.tensor([[1.]]),
    )
    return data


@pytest.mark.parametrize("src_levels, dest_levels, src_position, dest_position", [
    ([0, 0], [1, 0], 0, 1),      # a step along the array
    ([2, 3], [2, 4], 17, 22),    # a step around the rotation, given at the third array node of the fourth rotated copy
])
@pytest.mark.parametrize("replicate_levels", [None, [True, False], [False, True], [False, False]])
def test_set_edge_attr_by_pattern_level_matches_set_edge_attr_per_expansion_by_pattern(
    src_levels, dest_levels, src_position, dest_position, replicate_levels,
):

    by_level = build_rotated_array()
    by_level.set_edge_attr_by_pattern_level(
        "force", [0], [0], [src_levels], [dest_levels], torch.tensor([[7.]]), replicate_levels=replicate_levels,
    )

    per_expansion = build_rotated_array()
    per_expansion.set_edge_attr_per_expansion_by_pattern(
        "force", [0], [0], [src_position], [dest_position], torch.tensor([[7.]]), replicate_levels=replicate_levels,
    )

    assert torch.any(by_level.force == 7.)
    assert torch.equal(by_level.force, per_expansion.force)


def test_set_edge_attr_by_pattern_level_several_edges():

    value = torch.tensor([[5.], [6.]])

    by_level = build_rotated_array()
    by_level.set_edge_attr_by_pattern_level(
        "force", [0, 0], [0, 0], torch.tensor([[0, 0], [0, 0]]), torch.tensor([[1, 0], [0, 1]]), value,
    )

    per_expansion = build_rotated_array()
    per_expansion.set_edge_attr_per_expansion_by_pattern("force", [0, 0], [0, 0], [0, 0], [1, 5], value)

    assert sorted(set(by_level.force.view(-1).tolist())) == [5., 6.]
    assert torch.equal(by_level.force, per_expansion.force)


def test_set_edge_attr_by_pattern_level_at_a_merged_copy():

    data = build_mirrored_array()

    # a step along the array, given on the mirrored side: its start, [0, 1], is merged into [0, 0], from where the step
    # to [1, 1] would cross the mirror plane instead
    with pytest.raises(ValueError, match="an end of it was merged into another copy"):
        data.set_edge_attr_by_pattern_level("force", [0], [0], [[0, 1]], [[1, 1]], torch.tensor([[3.]]))

    assert torch.equal(data.force, torch.ones_like(data.force))

    # given on the side that was not merged, the same step sets the whole chain, on both sides
    data.set_edge_attr_by_pattern_level("force", [0], [0], [[0, 0]], [[1, 0]], torch.tensor([[3.]]))

    assert torch.equal(data.force, torch.full_like(data.force, 3.))


@pytest.mark.parametrize("src_expansion_ids, dest_expansion_ids, src_levels, dest_levels, value, message", [
    ([0, 0], [0], [[0, 0]], [[1, 0]], torch.ones(1, 1), "src_expansion_ids and dest_expansion_ids must have one entry per edge, got 2 and 1"),
    ([0], [0], [[0]], [[1, 0]], torch.ones(1, 1), r"src_level_positions must have shape \[1, 2\]"),
    ([0], [0], [[0, 0]], [[0, 12]], torch.ones(1, 1), r"level 1 has 12 positions \(0 to 11\), got 12"),
    ([3], [0], [[0, 0]], [[1, 0]], torch.ones(1, 1), r"No expansion found with id\(s\) \[3\]"),
    ([0], [0], [[0, 0]], [[2, 0]], torch.ones(1, 1), r"No edge found for \(src, dest\) pairs"),
    ([0], [0], [[0, 0]], [[1, 0]], torch.ones(2, 1), r"value must have one row per entry \(1\), but has 2"),
])
def test_set_edge_attr_by_pattern_level_rejects_invalid_input(src_expansion_ids, dest_expansion_ids, src_levels, dest_levels, value, message):

    data = build_rotated_array()
    force_before = data.force.clone()

    with pytest.raises(ValueError, match=message):
        data.set_edge_attr_by_pattern_level("force", src_expansion_ids, dest_expansion_ids, src_levels, dest_levels, value)

    # the failed call left the attribute unchanged
    assert torch.equal(data.force, force_before)


def test_set_edge_attr__level_without_pattern():

    data = StructData(
        node_attrs={"coords": torch.empty((0, 3), dtype=torch.float)},
        edge_attrs={"force": torch.empty((0, 1), dtype=torch.float)},
    )
    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [1., 0., 0.]]))
    data.add_edges(edge_indices=torch.tensor([[0], [1]]), force=torch.tensor([[1.]]))

    with pytest.raises(ValueError, match="No pattern is registered"):
        data.set_edge_attr_by_pattern_level("force", [0], [0], [[0]], [[1]], torch.tensor([[2.]]))


def test_set_edge_attr_by_pattern_level_without_edges():

    data = build_rotated_array()
    force_before = data.force.clone()

    data.set_edge_attr_by_pattern_level("force", [], [], [], [], torch.empty((0, 1)))

    assert torch.equal(data.force, force_before)
