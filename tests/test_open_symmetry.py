import pytest
import torch
from torch_structure.data.data import StructData


def build(symmetry, n_seeds=1):
    """A graph with one registered symmetry and ``n_seeds`` seed nodes expanded over it."""
    node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
    edge_attrs = {"force": torch.empty((0, 1), dtype=torch.float)}

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)
    data.add_symmetry({"s": symmetry}, copy_attrs=["force"])

    coords = torch.tensor([[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])[:n_seeds]
    data.add_nodes(symmetry="s", coords=coords)
    return data


def undirected_edges(data):
    """The canonical half of each reciprocal pair, as a sorted list of (low, high) tuples."""
    forward = data.directed_mask.view(-1)
    edge_index = data.edge_index[:, forward]
    return sorted(
        tuple(sorted((int(src), int(dst))))
        for src, dst in zip(edge_index[0], edge_index[1])
    )


def add_family(data, src_position, dest_position, orbit=0, force=1.0):
    data.add_edges_by_orbit(
        src_orbit_ids=[orbit], dest_orbit_ids=[orbit],
        src_orbit_positions=[src_position], dest_orbit_positions=[dest_position],
        force=torch.tensor([[force]]),
    )


# --------------------------------------------------------------------------- flat open rotation

def test_open_rotation_leaves_out_the_closing_edge():

    data = build(StructData().create_rotational_symmetry(4, closes=False))

    add_family(data, 0, 1)

    # the ring stays open: (3, 0) is not added
    assert undirected_edges(data) == [(0, 1), (1, 2), (2, 3)]


def test_closing_rotation_still_adds_the_closing_edge():

    data = build(StructData().create_rotational_symmetry(4))

    add_family(data, 0, 1)

    assert undirected_edges(data) == [(0, 1), (0, 3), (1, 2), (2, 3)]


def test_rotation_closes_by_default():

    rotation = StructData().create_rotational_symmetry(4)

    assert torch.equal(rotation["closes"], torch.ones(4, dtype=torch.bool))


def test_open_rotation_half_turn_is_unaffected():

    # a half turn never wraps past the end, so open and closing agree
    expected = [(0, 2), (1, 3)]

    for closes in (True, False):
        data = build(StructData().create_rotational_symmetry(4, closes=closes))
        add_family(data, 0, 2)
        assert undirected_edges(data) == expected


def test_open_rotation_keeps_an_explicitly_given_closing_edge():

    data = build(StructData().create_rotational_symmetry(4, closes=False))

    # the seed edge itself never runs off the end, so it survives on its own
    add_family(data, 3, 0)

    assert undirected_edges(data) == [(0, 3)]


def test_open_rotation_with_two_orbits():

    data = build(StructData().create_rotational_symmetry(3, closes=False), n_seeds=2)

    # orbit 0 is nodes 0-2, orbit 1 is nodes 3-5
    data.add_edges_by_orbit(
        src_orbit_ids=[0], dest_orbit_ids=[1],
        src_orbit_positions=[0], dest_orbit_positions=[1],
        force=torch.tensor([[1.]]),
    )

    # positions 0->1 and 1->2 across the two orbits, but not 2->0
    assert undirected_edges(data) == [(0, 4), (1, 5)]


# --------------------------------------------------------------------------- nested open rotations

def nested(inner_closes, outer_closes):
    """A 4-fold rotation (level 0) nested inside a 3-fold rotation (level 1)."""
    data = StructData()
    return data.combine_symmetry(
        data.create_rotational_symmetry(4, closes=inner_closes),
        data.create_rotational_symmetry(3, origin=torch.tensor([10., 0., 0.]), closes=outer_closes),
    )


# node index = outer_digit * 4 + inner_digit
INNER_RINGS_CLOSED = [
    (0, 1), (1, 2), (2, 3), (0, 3),
    (4, 5), (5, 6), (6, 7), (4, 7),
    (8, 9), (9, 10), (10, 11), (8, 11),
]
INNER_RINGS_OPEN = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
]


@pytest.mark.parametrize("outer_closes", [True, False])
def test_nested_inner_level_follows_the_inner_flag(outer_closes):

    # an edge inside one 4-fold ring is swept over both levels, so only the inner flag applies
    data = build(nested(inner_closes=False, outer_closes=outer_closes))
    add_family(data, 0, 1)
    assert undirected_edges(data) == sorted(INNER_RINGS_OPEN)

    data = build(nested(inner_closes=True, outer_closes=outer_closes))
    add_family(data, 0, 1)
    assert undirected_edges(data) == sorted(INNER_RINGS_CLOSED)


OUTER_RUNGS_OPEN = [(i, i + 4) for i in range(4)] + [(i + 4, i + 8) for i in range(4)]
OUTER_RUNGS_CLOSED = OUTER_RUNGS_OPEN + [(i, i + 8) for i in range(4)]


@pytest.mark.parametrize("inner_closes", [True, False])
def test_nested_outer_level_follows_the_outer_flag(inner_closes):

    # an edge between two 4-fold rings is copied to all four inner positions, so only the outer
    # flag decides whether the third rung back to the first ring is added
    data = build(nested(inner_closes=inner_closes, outer_closes=False))
    add_family(data, 0, 4)
    assert undirected_edges(data) == sorted(OUTER_RUNGS_OPEN)

    data = build(nested(inner_closes=inner_closes, outer_closes=True))
    add_family(data, 0, 4)
    assert undirected_edges(data) == sorted(OUTER_RUNGS_CLOSED)


def test_doubly_nested_open_is_open_in_both_directions():

    data = build(nested(inner_closes=False, outer_closes=False))

    add_family(data, 0, 1, force=1.0)   # along the inner level
    add_family(data, 0, 4, force=2.0)   # along the outer level

    expected = sorted(INNER_RINGS_OPEN + OUTER_RUNGS_OPEN)
    assert undirected_edges(data) == expected

    # neither level wrapped: no (3, 0)-style edge and no (0, 8)-style rung
    assert (0, 3) not in expected
    assert (0, 8) not in expected


# --------------------------------------------------------------------------- the stored flags

def test_closing_flag_sits_on_the_last_matrix_of_each_level():

    T, F = True, False
    expected = {
        (True, True): [T] * 12,
        (False, True): [T, T, T, F] + [T] * 8,
        (True, False): [T] * 11 + [F],
        (False, False): [T, T, T, F] + [T] * 7 + [F],
    }

    for (inner_closes, outer_closes), flags in expected.items():
        symmetry = nested(inner_closes, outer_closes)
        assert torch.equal(symmetry["closes"], torch.tensor(flags)), (inner_closes, outer_closes)


def test_registered_flags_decode_back_to_the_levels():

    for inner_closes in (True, False):
        for outer_closes in (True, False):
            data = build(nested(inner_closes, outer_closes))
            level_sizes, level_closes = data._infer_level_sizes(0)

            assert level_sizes == [4, 3]
            assert level_closes == [inner_closes, outer_closes]


def test_triple_nesting_keeps_one_flag_per_level():

    data = StructData()
    inner = data.create_rotational_symmetry(4, closes=False)
    middle = data.create_rotational_symmetry(3, origin=torch.tensor([10., 0., 0.]))
    outer = data.create_rotational_symmetry(2, origin=torch.tensor([20., 0., 0.]), closes=False)

    symmetry = data.combine_symmetry(data.combine_symmetry(inner, middle), outer)
    graph = build(symmetry)

    level_sizes, level_closes = graph._infer_level_sizes(0)

    assert level_sizes == [4, 3, 2]
    assert level_closes == [False, True, False]
    # the flags sit at cumprod(level_sizes) - 1
    assert torch.where(~symmetry["closes"])[0].tolist() == [3, 23]


def test_add_symmetry_rejects_a_flag_off_a_level_boundary():

    data = StructData()
    symmetry = data.create_rotational_symmetry(4)
    symmetry["closes"] = torch.tensor([True, False, True, True])

    with pytest.raises(ValueError, match="only be False at the last matrix"):
        data.add_symmetry({"bad": symmetry})


def test_add_symmetry_rejects_wrongly_sized_flags():

    data = StructData()
    symmetry = data.create_rotational_symmetry(4)
    symmetry["closes"] = torch.ones(3, dtype=torch.bool)

    with pytest.raises(ValueError, match="one entry per matrix"):
        data.add_symmetry({"bad": symmetry})


def test_symmetry_dict_without_flags_closes():

    rotation = StructData().create_rotational_symmetry(4)
    legacy = {"matrices": rotation["matrices"], "symmetry_level": rotation["symmetry_level"]}

    data = build(legacy)
    add_family(data, 0, 1)

    assert torch.equal(data.symmetry_closes, torch.ones(4, dtype=torch.bool))
    assert undirected_edges(data) == [(0, 1), (0, 3), (1, 2), (2, 3)]


# --------------------------------------------------------------------------- translation

def test_translation_is_open_by_default():

    symmetry = StructData().create_translation(4, torch.tensor([2., 0., 0.]))

    assert torch.equal(symmetry["closes"], torch.tensor([True, True, True, False]))
    assert torch.equal(symmetry["symmetry_level"], torch.zeros(4, dtype=torch.long))
    assert torch.equal(symmetry["matrices"][:, :3, 3], torch.tensor([
        [0., 0., 0.], [2., 0., 0.], [4., 0., 0.], [6., 0., 0.],
    ]))


def test_translation_builds_an_open_chain():

    data = build(StructData().create_translation(4, torch.tensor([1., 0., 0.])))

    add_family(data, 0, 1)

    assert undirected_edges(data) == [(0, 1), (1, 2), (2, 3)]
    assert torch.equal(
        data.coords[:, 0], torch.tensor([1., 2., 3., 4.])
    )


def test_translation_can_close():

    data = build(StructData().create_translation(4, torch.tensor([1., 0., 0.]), closes=True))

    add_family(data, 0, 1)

    assert undirected_edges(data) == [(0, 1), (0, 3), (1, 2), (2, 3)]


def open_grid():
    """3 positions along x (level 0) nested inside 2 positions along y (level 1), both open."""
    data = StructData()
    return data.combine_symmetry(
        data.create_translation(3, torch.tensor([1., 0., 0.])),
        data.create_translation(2, torch.tensor([0., 1., 0.])),
    )


def test_doubly_nested_translation_is_open_in_both_directions():

    graph = build(open_grid())

    # one seed per direction is enough: each is copied over the whole symmetry
    add_family(graph, 0, 1)   # along x
    add_family(graph, 0, 3)   # along y

    # the full 3 x 2 grid: 2 edges per row, 1 per column, nothing wrapping in either direction
    assert undirected_edges(graph) == [(0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5)]

    level_sizes, level_closes = graph._infer_level_sizes(0)
    assert level_sizes == [3, 2]
    assert level_closes == [False, False]


def test_outer_level_edge_reaches_every_inner_position():

    graph = build(open_grid())

    # one seed along y reaches all three columns, and does not wrap back to the first row
    add_family(graph, 0, 3)

    assert undirected_edges(graph) == [(0, 3), (1, 4), (2, 5)]


def test_translation_validates_its_arguments():

    data = StructData()

    with pytest.raises(ValueError, match="n must be at least 1"):
        data.create_translation(0)

    with pytest.raises(ValueError, match="must not be the zero vector"):
        data.create_translation(3, torch.zeros(3))

    with pytest.raises(ValueError, match="must have shape"):
        data.create_translation(3, torch.tensor([1., 0.]))


# --------------------------------------------------------------------------- per step scaling

def test_scaling_scales_each_step_individually():

    symmetry = StructData().create_translation(
        4, torch.tensor([1., 0., 0.]), scaling=torch.tensor([1., 2., 3.])
    )

    # the steps are 1, 2 and 3 long, so the positions sit at 0, 1, 3 and 6
    assert torch.equal(symmetry["matrices"][:, :3, 3], torch.tensor([
        [0., 0., 0.], [1., 0., 0.], [3., 0., 0.], [6., 0., 0.],
    ]))


def test_scaling_applies_to_the_whole_direction_vector():

    symmetry = StructData().create_translation(
        3, torch.tensor([2., 0., 1.]), scaling=torch.tensor([1., 0.5])
    )

    assert torch.equal(symmetry["matrices"][:, :3, 3], torch.tensor([
        [0., 0., 0.], [2., 0., 1.], [3., 0., 1.5],
    ]))


def test_scaling_of_ones_matches_the_default():

    data = StructData()
    default = data.create_translation(4, torch.tensor([2., 0., 0.]))
    explicit = data.create_translation(4, torch.tensor([2., 0., 0.]), scaling=torch.ones(3))

    assert torch.equal(default["matrices"], explicit["matrices"])


def test_scaling_accepts_a_non_float_dtype():

    symmetry = StructData().create_translation(
        3, torch.tensor([1., 0., 0.]), scaling=torch.tensor([2, 3])
    )

    assert torch.equal(symmetry["matrices"][:, :3, 3], torch.tensor([
        [0., 0., 0.], [2., 0., 0.], [5., 0., 0.],
    ]))


def test_scaling_may_be_negative():

    symmetry = StructData().create_translation(
        3, torch.tensor([1., 0., 0.]), scaling=torch.tensor([2., -1.])
    )

    assert torch.equal(symmetry["matrices"][:, :3, 3], torch.tensor([
        [0., 0., 0.], [2., 0., 0.], [1., 0., 0.],
    ]))


def test_scaling_keeps_the_closing_flag_and_level():

    symmetry = StructData().create_translation(
        4, torch.tensor([1., 0., 0.]), scaling=torch.tensor([1., 2., 3.]), closes=True
    )

    assert torch.equal(symmetry["closes"], torch.ones(4, dtype=torch.bool))
    assert torch.equal(symmetry["symmetry_level"], torch.zeros(4, dtype=torch.long))


def test_scaled_translation_builds_an_unevenly_spaced_chain():

    data = build(StructData().create_translation(
        4, torch.tensor([1., 0., 0.]), scaling=torch.tensor([1., 2., 3.])
    ))

    add_family(data, 0, 1)

    # the seed node sits at x = 1, so the chain runs 1, 2, 4, 7
    assert torch.equal(data.coords[:, 0], torch.tensor([1., 2., 4., 7.]))
    assert undirected_edges(data) == [(0, 1), (1, 2), (2, 3)]


def test_scaling_can_be_nested():

    data = StructData()
    symmetry = data.combine_symmetry(
        data.create_translation(3, torch.tensor([1., 0., 0.]), scaling=torch.tensor([1., 3.])),
        data.create_translation(2, torch.tensor([0., 1., 0.]), scaling=torch.tensor([5.])),
    )

    graph = build(symmetry)

    # x follows 0, 1, 4 within each row, y jumps by 5 between the two rows
    assert torch.equal(graph.coords[:, 0], torch.tensor([1., 2., 5., 1., 2., 5.]))
    assert torch.equal(graph.coords[:, 1], torch.tensor([0., 0., 0., 5., 5., 5.]))

    level_sizes, level_closes = graph._infer_level_sizes(0)
    assert level_sizes == [3, 2]
    assert level_closes == [False, False]


def test_scaling_must_have_one_entry_per_step():

    data = StructData()

    with pytest.raises(ValueError, match=r"scaling must have shape \[3\]"):
        data.create_translation(4, torch.tensor([1., 0., 0.]), scaling=torch.ones(4))

    with pytest.raises(ValueError, match=r"scaling must have shape \[0\]"):
        data.create_translation(1, torch.tensor([1., 0., 0.]), scaling=torch.ones(1))


def test_scaling_must_be_a_tensor():

    data = StructData()

    with pytest.raises(ValueError, match="scaling must be a torch.Tensor"):
        data.create_translation(3, torch.tensor([1., 0., 0.]), scaling=[1., 2.])


# --------------------------------------------------------------------------- setters see the same family

def test_set_edge_attr_follows_the_open_family():

    data = build(StructData().create_rotational_symmetry(4, closes=False))
    add_family(data, 0, 1, force=1.0)

    mask = torch.zeros(data.num_edges, dtype=torch.bool)
    mask[0] = True
    data.set_edge_attr("force", mask, torch.tensor([[7.]]))

    # all three edges of the open family are updated, both directions each
    assert torch.equal(data.force, torch.full((6, 1), 7.))


def test_set_edge_attr_by_orbit_follows_the_open_family():

    data = build(StructData().create_rotational_symmetry(4, closes=False))
    add_family(data, 0, 1, force=1.0)

    data.set_edge_attr_by_orbit(
        "force",
        src_orbit_ids=[0], dest_orbit_ids=[0],
        src_orbit_positions=[1], dest_orbit_positions=[2],
        value=torch.tensor([[5.]]),
    )

    assert torch.equal(data.force, torch.full((6, 1), 5.))


# --------------------------------------------------------------------------- mirrors are unchanged

def test_mirror_always_closes_and_adds_one_edge():

    mirror = StructData().create_mirror_symmetry()

    assert torch.equal(mirror["closes"], torch.ones(2, dtype=torch.bool))

    data = build(mirror)
    add_family(data, 0, 1)

    # the two copies are the same undirected edge, so only one is added
    assert undirected_edges(data) == [(0, 1)]


def test_open_level_inside_a_mirror():

    data = StructData()
    symmetry = data.combine_symmetry(
        data.create_translation(3, torch.tensor([1., 0., 0.])),
        data.create_mirror_symmetry(normal=torch.tensor([0., 1., 0.])),
    )

    graph = build(symmetry)
    add_family(graph, 0, 1)

    # an open chain of 3 on each side of the mirror
    assert undirected_edges(graph) == [(0, 1), (1, 2), (3, 4), (4, 5)]


# --------------------------------------------------------------------------- replicate_levels

def rot_mirror_rot():
    """A 4-fold rotation (level 0) inside a mirror (level 1) inside a 3-fold rotation (level 2)."""
    data = StructData()
    return data.combine_symmetry(
        data.combine_symmetry(
            data.create_rotational_symmetry(4),
            data.create_mirror_symmetry(normal=torch.tensor([0., 0., 1.])),
        ),
        data.create_rotational_symmetry(3, origin=torch.tensor([5., 0., 0.])),
    )


def seed_with(levels):
    data = build(rot_mirror_rot())
    data.add_edges_by_orbit(
        src_orbit_ids=[0], dest_orbit_ids=[0],
        src_orbit_positions=[0], dest_orbit_positions=[1],
        force=torch.tensor([[1.]]), replicate_levels=levels,
    )
    return data


@pytest.mark.parametrize("levels, expected", [
    ([True, True, True], 24),     # 4 * 2 * 3
    ([True, False, True], 12),    # 4 * 3, the mirror frozen between two replicated levels
    ([True, True, False], 8),     # 4 * 2
    ([False, True, True], 6),     # 2 * 3
    ([True, False, False], 4),    # 4
    ([False, False, False], 1),   # the given edge only
])
def test_replicate_levels_selects_the_levels(levels, expected):

    assert len(undirected_edges(seed_with(levels))) == expected


def test_replicate_levels_all_true_matches_the_default():

    data = build(rot_mirror_rot())
    add_family(data, 0, 1)

    assert undirected_edges(seed_with([True, True, True])) == undirected_edges(data)


def test_replicate_levels_none_matches_consider_symmetry_false():

    data = build(rot_mirror_rot())
    data.add_edges_by_orbit(
        src_orbit_ids=[0], dest_orbit_ids=[0],
        src_orbit_positions=[0], dest_orbit_positions=[1],
        force=torch.tensor([[1.]]), consider_symmetry=False,
    )

    assert undirected_edges(seed_with([False, False, False])) == undirected_edges(data)


def test_replicate_levels_keeps_a_frozen_level_on_the_seed_position():

    # the mirror is frozen, so every edge stays on the seed's own side of the mirror plane
    data = seed_with([True, False, True])
    positions = data.orbit_position.view(-1)

    for src, dst in undirected_edges(data):
        assert (int(positions[src]) // 4) % 2 == 0
        assert (int(positions[dst]) // 4) % 2 == 0


def test_replicate_levels_still_respects_the_closing_flag():

    data = StructData()
    symmetry = data.combine_symmetry(
        data.create_translation(4, direction=torch.tensor([1., 0., 0.])),
        data.create_rotational_symmetry(3),
    )
    graph = build(symmetry)
    graph.add_edges_by_orbit(
        src_orbit_ids=[0], dest_orbit_ids=[0],
        src_orbit_positions=[0], dest_orbit_positions=[1],
        force=torch.tensor([[1.]]), replicate_levels=[True, False],
    )

    # the open array gives 3 steps, not 4, and the frozen rotation keeps them in one sector
    assert undirected_edges(graph) == [(0, 1), (1, 2), (2, 3)]


def test_replicate_levels_applies_to_set_edge_attr():

    data = seed_with([True, False, True])

    mask = torch.zeros(data.num_edges, dtype=torch.bool)
    mask[0] = True
    data.set_edge_attr("force", mask, torch.tensor([[9.]]), replicate_levels=[True, False, True])

    assert torch.equal(data.force, torch.full((24, 1), 9.))


def test_replicate_levels_validates_its_length():

    data = build(rot_mirror_rot())

    with pytest.raises(ValueError, match="one entry per symmetry level"):
        data.add_edges_by_orbit(
            src_orbit_ids=[0], dest_orbit_ids=[0],
            src_orbit_positions=[0], dest_orbit_positions=[1],
            force=torch.tensor([[1.]]), replicate_levels=[True, False],
        )


def test_replicate_levels_still_deduplicates():

    # a mirror inside a rotation, replicated on the rotation only: 4 copies, not 8
    data = StructData()
    symmetry = data.combine_symmetry(
        data.create_mirror_symmetry(normal=torch.tensor([0., 1., 0.])),
        data.create_rotational_symmetry(4),
    )
    graph = build(symmetry)
    graph.add_edges_by_orbit(
        src_orbit_ids=[0], dest_orbit_ids=[0],
        src_orbit_positions=[0], dest_orbit_positions=[3],
        force=torch.tensor([[1.]]), replicate_levels=[False, True],
    )

    assert undirected_edges(graph) == [(0, 3), (1, 6), (2, 5), (4, 7)]


def test_set_edge_attr_raises_when_the_full_family_is_absent():

    data = build(rot_mirror_rot())
    data.add_edges_by_orbit(
        src_orbit_ids=[0], dest_orbit_ids=[0],
        src_orbit_positions=[0], dest_orbit_positions=[1],
        force=torch.zeros(1, 1), replicate_levels=[True, False, True],
    )

    mask = torch.zeros(data.num_edges, dtype=torch.bool)
    mask[0] = True

    # the default replicates on every level, but only the frozen family was ever added
    with pytest.raises(ValueError, match="No edge found for a symmetry copy"):
        data.set_edge_attr("force", mask, torch.tensor([[9.]]))


def test_replicate_levels_rejects_consider_symmetry_false():

    data = build(rot_mirror_rot())

    with pytest.raises(ValueError, match="cannot be combined with consider_symmetry=False"):
        data.add_edges_by_orbit(
            src_orbit_ids=[0], dest_orbit_ids=[0],
            src_orbit_positions=[0], dest_orbit_positions=[1],
            force=torch.tensor([[1.]]),
            consider_symmetry=False, replicate_levels=[True, True, True],
        )

    add_family(data, 0, 1)
    mask = torch.zeros(data.num_edges, dtype=torch.bool)
    mask[0] = True

    with pytest.raises(ValueError, match="cannot be combined with consider_symmetry=False"):
        data.set_edge_attr(
            "force", mask, torch.tensor([[9.]]),
            consider_symmetry=False, replicate_levels=[True, True, True],
        )


def test_replicate_levels_rejects_a_graph_without_symmetry():

    data = StructData(
        node_attrs={"coords": torch.empty((0, 3), dtype=torch.float)},
        edge_attrs={"force": torch.empty((0, 1), dtype=torch.float)},
    )
    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [1., 0., 0.]]))

    with pytest.raises(ValueError, match="no symmetry is registered"):
        data.add_edges(
            edge_indices=torch.tensor([[0], [1]]),
            force=torch.tensor([[1.]]), replicate_levels=[True, True, True],
        )

    data.add_edges(edge_indices=torch.tensor([[0], [1]]), force=torch.tensor([[1.]]))
    mask = torch.zeros(data.num_edges, dtype=torch.bool)
    mask[0] = True

    with pytest.raises(ValueError, match="no symmetry is registered"):
        data.set_edge_attr("force", mask, torch.tensor([[9.]]), replicate_levels=[True])
