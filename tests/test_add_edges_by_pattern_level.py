import pytest
import torch
from torch_structure.data.data import StructData


def new_data():
    node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
    edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}
    return StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)


def build_rotated_array():
    """An array of 5 along z (level 0) nested inside a 12-fold rotation (level 1), grown from one seed (expansion 0)."""
    data = new_data()
    rotated_array = data.nest_patterns([
        data.create_translational_pattern(5, direction=torch.tensor([0., 0., 1.])),
        data.create_rotational_pattern(12),
    ])
    data.add_patterns({"rotated_array": rotated_array})
    data.add_nodes_by_pattern(pattern="rotated_array", coords=torch.tensor([[1., 0., 0.]]))
    return data


def undirected_edges(data):
    """The canonical half of each reciprocal pair, as a sorted list of (low, high) tuples."""
    forward = data.directed_mask.view(-1)
    edge_index = data.edge_index[:, forward]
    return sorted(
        tuple(sorted((int(src), int(dst))))
        for src, dst in zip(edge_index[0], edge_index[1])
    )


@pytest.mark.parametrize("src_levels, dest_levels, src_position, dest_position", [
    ([0, 0], [1, 0], 0, 1),      # one step along the array
    ([0, 0], [0, 1], 0, 5),      # one step around the rotation
    ([2, 3], [3, 4], 17, 23),    # a diagonal, given at the third array node of the fourth rotated copy
])
def test_add_edges_by_pattern_level_matches_add_edges_per_expansion_by_pattern(src_levels, dest_levels, src_position, dest_position):

    by_level = build_rotated_array()
    by_level.add_edges_by_pattern_level(
        src_expansion_ids=[0], dest_expansion_ids=[0],
        src_level_positions=[src_levels], dest_level_positions=[dest_levels],
        force=torch.tensor([[1.]]),
    )

    per_expansion = build_rotated_array()
    per_expansion.add_edges_per_expansion_by_pattern(
        src_expansion_ids=[0], dest_expansion_ids=[0],
        src_expansion_positions=[src_position], dest_expansion_positions=[dest_position],
        force=torch.tensor([[1.]]),
    )

    assert torch.equal(by_level.edge_index, per_expansion.edge_index)
    assert torch.equal(by_level.force, per_expansion.force)


def test_add_edges_by_pattern_level_several_edges_and_replicate_levels():

    data = build_rotated_array()

    # two edges at once, replicated along the array only, so both stay between the first two rotated copies: the
    # steps up the array in the first copy, and the step to the second copy at every array node
    data.add_edges_by_pattern_level(
        src_expansion_ids=[0, 0], dest_expansion_ids=[0, 0],
        src_level_positions=torch.tensor([[0, 0], [0, 0]]), dest_level_positions=torch.tensor([[1, 0], [0, 1]]),
        force=torch.tensor([[1.], [2.]]), replicate_levels=[True, False],
    )

    assert undirected_edges(data) == [(0, 1), (0, 5), (1, 2), (1, 6), (2, 3), (2, 7), (3, 4), (3, 8), (4, 9)]


def test_add_edges_by_pattern_level_validates_replicate_levels():

    data = build_rotated_array()

    with pytest.raises(ValueError, match="one entry per pattern level"):
        data.add_edges_by_pattern_level(
            src_expansion_ids=[0], dest_expansion_ids=[0],
            src_level_positions=[[0, 0]], dest_level_positions=[[1, 0]],
            force=torch.tensor([[1.]]), replicate_levels=[True],
        )


def build_mirrored_truss():

    data = new_data()
    truss = data.nest_patterns([
        data.create_translational_pattern(4, torch.tensor([1., 0., 0.])),
        data.create_mirror_pattern(normal=torch.tensor([1., 0., 0.])),
        data.create_mirror_pattern(origin=torch.tensor([0., 0., 1.]), normal=torch.tensor([0., 0., 1.])),
    ])
    data.add_patterns({"truss": truss})
    data.add_nodes_by_pattern(pattern="truss", coords=torch.tensor([[0., 0., 0.]]))
    return data


def test_add_edges_by_pattern_level_refuses_a_merged_end_that_changes_the_edge():

    data = build_mirrored_truss()

    # a step along the array on the mirrored side: its start, [0, 1, 0], is merged into [0, 0, 0], from where the
    # step to [1, 1, 0] would cross the mirror plane instead
    with pytest.raises(ValueError, match="an end of it was merged into another copy"):
        data.add_edges_by_pattern_level(
            src_expansion_ids=[0], dest_expansion_ids=[0],
            src_level_positions=[[0, 1, 0]], dest_level_positions=[[1, 1, 0]],
            force=torch.tensor([[1.]]),
        )

    assert data.num_edges == 0

    # given on the side that was not merged, the same step adds both chords, joined at the merged nodes
    data.add_edges_by_pattern_level(
        src_expansion_ids=[0], dest_expansion_ids=[0],
        src_level_positions=[[0, 0, 0]], dest_level_positions=[[1, 0, 0]],
        force=torch.tensor([[1.]]),
    )

    x, z = data.coords[:, 0], data.coords[:, 2]
    for src, dst in undirected_edges(data):
        assert abs(x[src] - x[dst]) == 1. and z[src] == z[dst]
    assert len(undirected_edges(data)) == 12


def test_add_edges_by_pattern_level_accepts_a_merged_end_that_keeps_the_edge():

    data = build_mirrored_truss()

    # a vertical given on the mirrored side: both ends are merged, but the step from [0, 0, 0] to [0, 0, 1] is the
    # same, so it adds the same verticals, one at every x
    data.add_edges_by_pattern_level(
        src_expansion_ids=[0], dest_expansion_ids=[0],
        src_level_positions=[[0, 1, 0]], dest_level_positions=[[0, 1, 1]],
        force=torch.tensor([[1.]]),
    )

    x, z = data.coords[:, 0], data.coords[:, 2]
    verticals = undirected_edges(data)
    assert sorted(x[src].item() for src, dst in verticals) == [-3., -2., -1., 0., 1., 2., 3.]
    for src, dst in verticals:
        assert x[src] == x[dst] and z[src] != z[dst]


@pytest.mark.parametrize("src_expansion_ids, dest_expansion_ids, src_level_positions, dest_level_positions, message", [
    ([0], [0], [[0]], [[1, 0]], r"src_level_positions must have shape \[1, 2\]"),
    ([0], [0], [0, 0], [[1, 0]], r"src_level_positions must have shape \[1, 2\]"),
    ([0], [0], [[0, 0], [1, 0]], [[1, 0]], r"src_level_positions must have shape \[1, 2\]"),
    ([0], [0], [[0, 0]], [[1, 0, 0]], r"dest_level_positions must have shape \[1, 2\]"),
    ([0, 0], [0, 0], [[0, 0], [1]], [[1, 0], [2, 0]], "all of the same length"),
    ([0], [0], [[0., 0.]], [[1, 0]], "must hold integers"),
    ([0], [0], [[5, 0]], [[1, 0]], r"level 0 has 5 positions \(0 to 4\), got 5"),
    ([0], [0], [[0, 0]], [[0, -1]], r"level 1 has 12 positions \(0 to 11\), got -1"),
    ([0, 0], [0], [[0, 0]], [[1, 0]], "must have one entry per edge, got 2 and 1"),
    ([3], [0], [[0, 0]], [[1, 0]], r"No expansion found with id\(s\) \[3\]"),
])
def test_add_edges_by_pattern_level_rejects_invalid_input(src_expansion_ids, dest_expansion_ids, src_level_positions, dest_level_positions, message):

    data = build_rotated_array()

    with pytest.raises(ValueError, match=message):
        data.add_edges_by_pattern_level(
            src_expansion_ids=src_expansion_ids, dest_expansion_ids=dest_expansion_ids,
            src_level_positions=src_level_positions, dest_level_positions=dest_level_positions,
            force=torch.ones(len(src_expansion_ids), 1),
        )

    # the failed call left the graph unchanged
    assert data.num_edges == 0


def test_add_edges_by_pattern_level_rejects_expansions_of_several_patterns():

    data = new_data()
    data.add_patterns({"rot4": data.create_rotational_pattern(4), "mirror": data.create_mirror_pattern()})

    # expansion 0: the rotation, expansion 1: the mirror
    data.add_nodes_by_pattern(pattern="rot4", coords=torch.tensor([[1., 0., 0.]]))
    data.add_nodes_by_pattern(pattern="mirror", coords=torch.tensor([[3., 1., 0.]]))

    with pytest.raises(ValueError, match="several registered patterns"):
        data.add_edges_by_pattern_level(
            src_expansion_ids=[0], dest_expansion_ids=[1],
            src_level_positions=[[0]], dest_level_positions=[[0]],
            force=torch.tensor([[1.]]),
        )


def test_add_edges_by_pattern_level_without_pattern():

    data = new_data()
    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [1., 0., 0.]]))

    with pytest.raises(ValueError, match="No pattern is registered"):
        data.add_edges_by_pattern_level(
            src_expansion_ids=[0], dest_expansion_ids=[0],
            src_level_positions=[[0]], dest_level_positions=[[1]],
            force=torch.tensor([[1.]]),
        )


def test_add_edges_by_pattern_level_passes_on_errors_of_add_edges_by_pattern():

    data = build_rotated_array()
    data.add_edges_by_pattern_level(
        src_expansion_ids=[0], dest_expansion_ids=[0],
        src_level_positions=[[0, 0]], dest_level_positions=[[1, 0]],
        force=torch.tensor([[1.]]),
    )

    # the same edge family again, given at another copy
    with pytest.raises(ValueError, match="already exist"):
        data.add_edges_by_pattern_level(
            src_expansion_ids=[0], dest_expansion_ids=[0],
            src_level_positions=[[2, 7]], dest_level_positions=[[3, 7]],
            force=torch.tensor([[1.]]),
        )


def test_add_edges_by_pattern_level_without_edges():

    data = build_rotated_array()
    data.add_edges_by_pattern_level(
        src_expansion_ids=[], dest_expansion_ids=[],
        src_level_positions=[], dest_level_positions=[],
        force=torch.empty((0, 1)),
    )

    assert data.num_edges == 0
