import warnings

import pytest
import torch
from torch_structure.data.data import StructData


def build(pattern, transform_attrs=None):
    """A graph with coords and load node attributes, a force edge attribute and one registered pattern."""
    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    }
    edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)
    data.add_patterns({"s": pattern(data)}, transform_attrs=transform_attrs)
    return data


def mirrored_array(data):
    """An array of 4 along x, mirrored at the plane x=0."""
    x = torch.tensor([1., 0., 0.])
    return data.nest_patterns([data.create_translational_pattern(4, x), data.create_mirror_pattern(normal=x)])


def undirected_edges(data):
    """The canonical half of each reciprocal pair, as a sorted list of (low, high) tuples."""
    forward = data.directed_mask.view(-1)
    edge_index = data.edge_index[:, forward]
    return sorted(
        tuple(sorted((int(src), int(dst))))
        for src, dst in zip(edge_index[0], edge_index[1])
    )


def test_copies_on_a_mirror_plane_are_merged():

    data = build(mirrored_array)

    # the array starts on the mirror plane, so position 4, the mirror image of position 0, lands on it and is merged
    # into it
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[0., 0., 0.]]))

    assert data.num_nodes == 7
    assert data.coords[:, 0].tolist() == [0., 1., 2., 3., -1., -2., -3.]
    assert data.expansion_position.view(-1).tolist() == [0, 1, 2, 3, 5, 6, 7]
    assert data.expansion_holes.tolist() == [[0, 4, 0]]


def test_edge_copies_attach_to_the_merged_node():

    data = build(mirrored_array)
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[0., 0., 0.]]))

    # a step along the array; its mirrored copy starts at the merged position 4 and so at node 0, which joins the two
    # halves
    data.add_edges_by_pattern(edge_indices=torch.tensor([[0], [1]]), force=torch.tensor([[1.]]))

    chain = [(0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6)]
    assert undirected_edges(data) == chain

    # an edge across the plane: its copy at the merged node would join node 0 to itself and is left out
    data.add_edges_by_pattern(edge_indices=torch.tensor([[1], [4]]), force=torch.tensor([[2.]]))

    rungs = [(1, 4), (2, 5), (3, 6)]
    assert undirected_edges(data) == sorted(chain + rungs)

    # setting one edge of the chain sets the whole chain, the edge at the merged node included
    mask = (data.edge_index[0] == 0) & (data.edge_index[1] == 1)
    data.set_edge_attr_by_pattern("force", mask, torch.tensor([[5.]]))

    for (src, dst), force in zip(data.edge_index.t().tolist(), data.force.view(-1).tolist()):
        assert force == (5. if tuple(sorted((src, dst))) in chain else 2.)


def test_copies_on_a_rotation_axis_are_merged():

    data = build(lambda d: d.create_rotational_pattern(6))

    # the apex lies on the rotation axis, so all its 6 copies are one node; the ring is not affected
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[0., 0., 2.], [1., 0., 0.]]))

    assert data.num_nodes == 7
    assert data.expansion_holes.tolist() == [[0, position, 0] for position in range(1, 6)]

    # one spoke given, all 6 added, every one of them from the single apex node
    data.add_edges_by_pattern(edge_indices=torch.tensor([[0], [1]]), force=torch.tensor([[1.]]))

    assert undirected_edges(data) == [(0, node) for node in range(1, 7)]


def test_overlapping_patterns_are_merged():

    data = build(lambda d: d.nest_patterns([d.create_rotational_pattern(4), d.create_rotational_pattern(6)]))

    # a 4-fold inside a 6-fold rotation about the same axis repeats every copy twice: 24 positions, 12 nodes 30
    # degrees apart
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[1., 0., 0.]]))

    assert data.num_nodes == 12
    assert data.expansion_holes.shape == (12, 3)

    # a 60 degree step from position 0 to position 4: every copy, also those starting or ending at a merged position,
    # joins a node to the one 60 degrees on
    data.add_edges_by_pattern(edge_indices=torch.tensor([[0], [4]]), force=torch.tensor([[1.]]))

    edges = undirected_edges(data)
    assert len(edges) == 12

    for src, dst in edges:
        cos = torch.dot(data.coords[src], data.coords[dst]) / (data.coords[src].norm() * data.coords[dst].norm())
        assert torch.isclose(cos, torch.tensor(0.5), atol=1e-5)


def test_set_node_attr_by_pattern_on_a_merged_node():

    data = build(mirrored_array)
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[0., 0., 0.]]))

    # lifting node 1 lifts the whole array; node 0 is reached through positions 0 and 4, which agree on it
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[1] = True
    data.set_node_attr_by_pattern("coords", mask, torch.tensor([[1., 1., 0.]]))

    coords_expected = torch.tensor([[0., 1., 0.], [1., 1., 0.], [2., 1., 0.], [3., 1., 0.], [-1., 1., 0.], [-2., 1., 0.], [-3., 1., 0.]])
    assert torch.allclose(data.coords, coords_expected)

    # moving node 0 off the plane would put positions 0 and 4 in different places
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[0] = True
    with pytest.raises(ValueError, match="were merged into one node"):
        data.set_node_attr_by_pattern("coords", mask, torch.tensor([[0.5, 1., 0.]]))

    # the failed call left the attribute unchanged
    assert torch.allclose(data.coords, coords_expected)


def test_set_node_attr_by_pattern_on_a_node_on_the_rotation_axis():

    data = build(lambda d: d.create_rotational_pattern(6))
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[0., 0., 2.]]))

    mask = torch.ones(1, dtype=torch.bool)

    # along the axis every copy agrees, off it they do not
    data.set_node_attr_by_pattern("coords", mask, torch.tensor([[0., 0., 3.]]))
    assert torch.allclose(data.coords, torch.tensor([[0., 0., 3.]]))

    with pytest.raises(ValueError, match="were merged into one node"):
        data.set_node_attr_by_pattern("coords", mask, torch.tensor([[0.5, 0., 3.]]))


def test_per_expansion_resolves_a_merged_position():

    data = build(mirrored_array)
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[0., 0., 0.]]))

    # position 4 was merged into position 0, so it addresses node 0
    data.set_node_attr_per_expansion("load", [0], [4], torch.tensor([[0., 0., -1.]]))
    assert data.load[0].tolist() == [0., 0., -1.]

    data.add_edges_per_expansion(
        src_expansion_ids=[0], dest_expansion_ids=[0], src_expansion_positions=[4], dest_expansion_positions=[6],
        force=torch.tensor([[1.]]),
    )
    assert undirected_edges(data) == [(0, 5)]


def test_coinciding_copies_must_agree_in_every_attribute():

    data = build(lambda d: d.create_mirror_pattern(), transform_attrs=["coords", "load"])

    # the seed lies on the mirror plane, but its load points across it, so its two copies cannot be one node
    with pytest.raises(ValueError, match="In expansion 0, positions 0 and 1 would be merged into one node, but their 'load' differs"):
        data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[0., 1., 0.]]), load=torch.tensor([[1., 0., 0.]]))

    # merge raises before it changes anything, but after the copies were added, so they stay, unmerged
    assert data.num_nodes == 2

    # a load along the plane is the same for both copies
    data = build(lambda d: d.create_mirror_pattern(), transform_attrs=["coords", "load"])
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[0., 1., 0.]]), load=torch.tensor([[0., -1., 0.]]))
    assert data.num_nodes == 1


def test_merge_attr():

    # without merging, both copies of a seed on the mirror plane stay
    data = build(lambda d: d.create_mirror_pattern(), transform_attrs=["coords", "load"])
    data.add_nodes_by_pattern(pattern="s", merge_attr=None, coords=torch.tensor([[0., 1., 0.]]))
    assert data.num_nodes == 2

    # merging by load instead: a seed on the plane with a load along it has two copies with the same load and coords
    data = build(lambda d: d.create_mirror_pattern(), transform_attrs=["coords", "load"])
    data.add_nodes_by_pattern(
        pattern="s", merge_attr="load", coords=torch.tensor([[0., 1., 0.]]), load=torch.tensor([[0., -1., 0.]]),
    )
    assert data.num_nodes == 1

    # off the plane, the two copies still share their load, but not their coords, so they cannot be merged by load
    data = build(lambda d: d.create_mirror_pattern(), transform_attrs=["coords", "load"])
    with pytest.raises(ValueError, match="positions 0 and 1 would be merged into one node, but their 'coords' differs"):
        data.add_nodes_by_pattern(
            pattern="s", merge_attr="load", coords=torch.tensor([[1., 1., 0.]]), load=torch.tensor([[0., -1., 0.]]),
        )
    assert data.num_nodes == 2

    # an attribute the pattern copies unchanged is the same for every copy, so it cannot be used
    data = build(lambda d: d.create_mirror_pattern())
    with pytest.raises(ValueError, match="merge_attr 'load' is not a transform attribute"):
        data.add_nodes_by_pattern(pattern="s", merge_attr="load", coords=torch.tensor([[1., 0., 0.]]))
    assert data.num_nodes == 0


def test_copies_of_two_seeds_on_the_same_spot_raise():

    data = build(lambda d: d.create_mirror_pattern())

    # the two seeds are mirror images of each other, so each one's mirror copy lands on the other seed
    with pytest.raises(ValueError, match="Expansion 0, position 0 and expansion 1, position 1 would be merged into one node, but a node of one expansion cannot be merged into a node of another"):
        data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[1., 0., 0.], [-1., 0., 0.]]))


def test_untransformed_coords_are_not_merged():

    # coords the pattern does not transform are equal for every copy, which does not make the copies one node
    data = build(lambda d: d.create_mirror_pattern(), transform_attrs=[])
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[1., 0., 0.]]))
    assert data.num_nodes == 2

    # the same for coords that are not given and so default to the same value for every copy
    data = build(lambda d: d.create_rotational_pattern(4))
    data.add_nodes_by_pattern(pattern="s", load=torch.tensor([[0., 0., -1.]]))
    assert data.num_nodes == 4


def test_merged_copies_keep_existing_edges():

    data = build(lambda d: d.create_rotational_pattern(6))
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[1., 0., 0.]]))
    data.add_edges_by_pattern(edge_indices=torch.tensor([[0], [1]]), force=torch.tensor([[1.]]))

    # the apex's copies are merged by merge, which renumbers the nodes and moves the ring's edges along with them
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[0., 0., 2.]]))

    assert data.num_nodes == 7
    edges = undirected_edges(data)
    assert len(edges) == 6
    for src, dst in edges:
        assert data.coords[src, 2] == 0. and data.coords[dst, 2] == 0.
        assert torch.isclose(torch.dist(data.coords[src], data.coords[dst]), torch.tensor(1.), atol=1e-5)


def test_merge_records_merged_nodes_of_an_expansion():

    data = build(lambda d: d.nest_patterns([
        d.create_translational_pattern(4, torch.tensor([1., 0., 0.])),
        d.create_mirror_pattern(origin=torch.tensor([5., 0., 0.])),
    ]))

    # the seed lies on the mirror plane x=5, but add_nodes_by_pattern is told not to merge its mirror copy
    data.add_nodes_by_pattern(pattern="s", merge_attr=None, coords=torch.tensor([[5., 0., 0.]]))
    assert data.num_nodes == 8

    # merging by coords afterwards puts that copy, position 4, into position 0
    data.merge_by_attribute("coords")

    assert data.num_nodes == 7
    assert data.expansion_holes.tolist() == [[0, 4, 0]]

    # a step along the array: its mirrored copy starts at position 4 and so at the kept node, joining the two halves
    data.add_edges_per_expansion_by_pattern(
        src_expansion_ids=[0], dest_expansion_ids=[0], src_expansion_positions=[0], dest_expansion_positions=[1],
        force=torch.tensor([[1.]]),
    )

    x = data.coords[:, 0]
    chain = sorted((round(min(x[a], x[b]).item()), round(max(x[a], x[b]).item())) for a, b in undirected_edges(data))
    assert chain == [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]


def test_merge_redirects_positions_merged_into_a_removed_node():

    data = build(lambda d: d.nest_patterns([
        d.create_mirror_pattern(),
        d.create_mirror_pattern(normal=torch.tensor([0., 1., 0.])),
    ]))

    # the seed lies on the plane x=0, so positions 1 and 3 are merged into positions 0 and 2
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[0., 0.5, 0.]]))
    assert data.expansion_holes.tolist() == [[0, 1, 0], [0, 3, 2]]

    # merging the two remaining nodes moves position 2 into position 0, and position 3 along with it; they sit at
    # y=0.5 and y=-0.5, so only the survivor's coords are kept
    with pytest.warns(UserWarning, match=r"Merged nodes differ in \['coords'\]"):
        data.merge(torch.zeros(data.num_nodes, dtype=torch.long))

    assert data.num_nodes == 1
    assert sorted(data.expansion_holes.tolist()) == [[0, 1, 0], [0, 2, 0], [0, 3, 0]]

    data.set_node_attr_per_expansion("load", [0], [3], torch.tensor([[0., 0., -1.]]))
    assert data.load.tolist() == [[0., 0., -1.]]


def test_merging_nodes_of_two_expansions_raises():

    data = build(lambda d: d.create_rotational_pattern(6))

    # expansion 0: the ring; expansion 1: the apex, its positions 1-5 merged into position 0
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[1., 0., 0.], [0., 0., 2.]]))

    expansion_id = data.expansion_id.view(-1)
    merge_group_id = torch.full((data.num_nodes,), -1)
    merge_group_id[torch.where(expansion_id == 1)[0]] = 0
    merge_group_id[torch.where(expansion_id == 0)[0][0]] = 0

    with pytest.raises(ValueError, match="a node of one expansion cannot be merged into a node of another"):
        data.merge(merge_group_id)

    # the failed call left the graph unchanged
    assert data.num_nodes == 7
    assert data.expansion_holes.shape == (5, 3)


def build_ranked(rank_outside):
    """A 4-fold rotation of one seed with rank 1, and a node outside any pattern, with the given rank, on top of the
    seed. The node outside comes first, so it has the lower index."""
    data = StructData(node_attrs={
        "coords": torch.empty((0, 3), dtype=torch.float),
        "rank": torch.empty((0, 1), dtype=torch.float),
    })
    data.add_patterns({"rot4": data.create_rotational_pattern(4)})
    data.add_nodes(coords=torch.tensor([[1., 0., 0.]]), rank=torch.tensor([[rank_outside]]))
    data.add_nodes_by_pattern(pattern="rot4", coords=torch.tensor([[1., 0., 0.]]), rank=torch.tensor([[1.]]))
    return data


def merge_seed_with_node_outside(data, **kwargs):
    merge_group_id = torch.full((data.num_nodes,), -1)
    merge_group_id[[0, 1]] = 0
    data.merge(merge_group_id, **kwargs)


def test_merge_keeps_the_expansion_node_over_a_node_outside_any_pattern():

    data = build_ranked(rank_outside=1.)

    # the node outside the pattern has the lower index, but the seed survives; both agree in every attribute, so
    # there is no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        merge_seed_with_node_outside(data)

    assert data.num_nodes == 4
    assert data.expansion_id.view(-1).tolist() == [0, 0, 0, 0]
    assert data.rank.view(-1).tolist() == [1., 1., 1., 1.]
    # the removed node was outside any pattern, so no expansion position was merged
    assert data.expansion_holes.shape == (0, 3)


def test_merge_warns_when_the_priority_prefers_a_node_outside_any_pattern():

    data = build_ranked(rank_outside=5.)

    with pytest.warns(UserWarning, match="Node 0, outside any pattern, has the highest 'rank' among the nodes merged into expansion 0, position 0"):
        merge_seed_with_node_outside(data, node_priority_attribute="rank")

    # the seed survived anyway
    assert data.rank.view(-1).tolist() == [1., 1., 1., 1.]

    # with equal ranks, the priority does not prefer the node outside the pattern
    data = build_ranked(rank_outside=1.)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        merge_seed_with_node_outside(data, node_priority_attribute="rank")

    assert data.expansion_id.view(-1).tolist() == [0, 0, 0, 0]
