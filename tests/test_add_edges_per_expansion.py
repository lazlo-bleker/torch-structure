import pytest
import torch
from torch_structure.data.data import StructData


def build_structure(with_pattern=True, nodes_by_pattern=True):

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    if with_pattern:
        data.add_patterns({"rot4": data.create_rotational_pattern(4)})

    if nodes_by_pattern:
        # expansion 0 (nodes 0-3) and expansion 1 (nodes 4-7)
        data.add_nodes_by_pattern(pattern="rot4", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))

    return data


def test_add_edges_per_expansion_by_pattern_all_copies():

    data = build_structure()

    # node 8 is outside any pattern and must not be mistaken for expansion 1, position 3
    data.add_nodes(coords=torch.tensor([[0., 0., 3.]]))

    data.add_edges_per_expansion_by_pattern(
        src_expansion_ids=[0], dest_expansion_ids=[1], src_expansion_positions=[3], dest_expansion_positions=[3],
        force=torch.tensor([[2.]]),
    )

    # all 4 copies are added, whichever expansion position was given
    edge_index_expected = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 0, 1, 2, 3]])
    force_expected = torch.full((8, 1), 2.)

    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.force, force_expected)


def test_add_edges_per_expansion_without_pattern():

    data = build_structure()

    # expansion 0, position 0 -> expansion 0, position 1 and expansion 0, position 2 -> expansion 1, position 3; add_edges_per_expansion
    # only adds these two edges
    data.add_edges_per_expansion(
        src_expansion_ids=[0, 0], dest_expansion_ids=[0, 1], src_expansion_positions=[0, 2], dest_expansion_positions=[1, 3],
        force=torch.tensor([[1.], [2.]]),
    )

    edge_index_expected = torch.tensor([[0, 2, 1, 7], [1, 7, 0, 2]])
    force_expected = torch.tensor([[1.], [2.], [1.], [2.]])

    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.force, force_expected)


@pytest.mark.parametrize("method", ["add_edges_per_expansion", "add_edges_per_expansion_by_pattern"])
def test_add_edges_per_expansion_invalid_input(method):

    data = build_structure()

    # expansion 5 does not exist
    with pytest.raises(ValueError, match="No node found"):
        getattr(data, method)([5], [0], [0], [0])

    # position 4 does not exist in a 4-fold pattern
    with pytest.raises(ValueError, match="No node found"):
        getattr(data, method)([0], [1], [0], [4])

    # a pattern is registered, but no node belongs to an expansion
    data = build_structure(nodes_by_pattern=False)
    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [1., 0., 0.]]))

    with pytest.raises(ValueError, match="No node found"):
        getattr(data, method)([0], [0], [0], [1])

    # no pattern registered
    data = build_structure(with_pattern=False, nodes_by_pattern=False)
    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [1., 0., 0.]]))

    with pytest.raises(AttributeError):
        getattr(data, method)([0], [0], [0], [1])


def build_mirrored_arrays():
    """Two arrays of 4 along x, at y=0 (expansion 0) and y=1 (expansion 1), mirrored at the plane x=0, which both start on. In
    both expansions, position 4, the mirror image of position 0, is merged into it."""
    data = StructData(
        node_attrs={"coords": torch.empty((0, 3), dtype=torch.float)},
        edge_attrs={"force": torch.empty((0, 1), dtype=torch.float)},
    )
    x = torch.tensor([1., 0., 0.])
    data.add_patterns({"s": data.nest_patterns([data.create_translational_pattern(4, x), data.create_mirror_pattern(normal=x)])})
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[0., 0., 0.], [0., 1., 0.]]))
    return data


def test_add_edges_per_expansion_by_pattern_refuses_a_merged_end_that_changes_the_edge():

    data = build_mirrored_arrays()

    # a step along the array, given on the mirrored side: from position 0, which position 4 was merged into, the step
    # to position 5 would cross the mirror plane instead
    with pytest.raises(ValueError, match="an end of it was merged into another copy"):
        data.add_edges_per_expansion_by_pattern(
            src_expansion_ids=[0], dest_expansion_ids=[0], src_expansion_positions=[4], dest_expansion_positions=[5],
            force=torch.tensor([[1.]]),
        )

    assert data.num_edges == 0

    # given where nothing was merged, the same step adds the chain on both sides
    data.add_edges_per_expansion_by_pattern(
        src_expansion_ids=[0], dest_expansion_ids=[0], src_expansion_positions=[0], dest_expansion_positions=[1],
        force=torch.tensor([[1.]]),
    )

    assert data.num_edges // 2 == 6


def test_add_edges_per_expansion_by_pattern_accepts_merged_ends_that_keep_the_edge():

    data = build_mirrored_arrays()

    # both ends were merged the same way, so the rung between the arrays reads the same from positions 0 and 0
    data.add_edges_per_expansion_by_pattern(
        src_expansion_ids=[0], dest_expansion_ids=[1], src_expansion_positions=[4], dest_expansion_positions=[4],
        force=torch.tensor([[1.]]),
    )

    forward = data.directed_mask.view(-1)
    src, dst = data.edge_index[:, forward]
    assert torch.equal(data.coords[src, 0], data.coords[dst, 0])
    assert sorted(data.coords[src, 0].tolist()) == [-3., -2., -1., 0., 1., 2., 3.]
