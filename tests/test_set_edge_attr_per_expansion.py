import pytest
import torch
from torch_structure.data.data import StructData


def build_structure(nodes_by_pattern=True):

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    data.add_patterns({"rot4": data.create_rotational_pattern(4)})

    if nodes_by_pattern:
        # expansion 0 (nodes 0-3) and expansion 1 (nodes 4-7)
        data.add_nodes_by_pattern(pattern="rot4", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))

        # radial edges: rows 0-3 (0 -> 4, 1 -> 5, 2 -> 6, 3 -> 7), ring edges: rows 4-7 (0 -> 1, 1 -> 2, 2 -> 3,
        # 3 -> 0), reciprocal rows 8-15
        data.add_edges_by_pattern(edge_indices=torch.tensor([[0, 0], [4, 1]]), force=torch.tensor([[3.], [-1.]]))

    return data


def test_set_edge_attr_per_expansion_by_pattern_propagates():

    data = build_structure()

    # node 8 is outside any pattern and must not be mistaken for expansion 1, position 3
    data.add_nodes(coords=torch.tensor([[0., 0., 3.]]))

    # the radial edge from expansion 0, position 3 to expansion 1, position 3
    data.set_edge_attr_per_expansion_by_pattern(
        "force", src_expansion_ids=[0], dest_expansion_ids=[1], src_expansion_positions=[3], dest_expansion_positions=[3],
        value=torch.tensor([[6.]]),
    )

    # all radial edges change, the ring edges do not
    force_expected = torch.tensor([[6.]] * 4 + [[-1.]] * 4 + [[6.]] * 4 + [[-1.]] * 4)

    assert torch.equal(data.force, force_expected)


def test_set_edge_attr_per_expansion_reversed_direction_and_without_pattern():

    data = build_structure()

    # expansion 1, position 1 -> expansion 0, position 1 is the reciprocal row of the radial edge 1 -> 5
    data.set_edge_attr_per_expansion_by_pattern("force", [1], [0], [1], [1], torch.tensor([[7.]]))

    # set_edge_attr_per_expansion only changes the ring edge 0 -> 1 (row 4) and its reciprocal row
    data.set_edge_attr_per_expansion("force", [0], [0], [0], [1], torch.tensor([[4.]]))

    force_expected = torch.tensor(
        [[7.]] * 4 + [[4.], [-1.], [-1.], [-1.]] + [[7.]] * 4 + [[4.], [-1.], [-1.], [-1.]]
    )

    assert torch.equal(data.force, force_expected)


def test_set_edge_attr_per_expansion_by_pattern_with_replicate_levels():

    data = build_structure()

    # rot4 has a single level: frozen, only the radial edge 3 -> 7 (row 3) and its reciprocal row change
    data.set_edge_attr_per_expansion_by_pattern(
        "force", src_expansion_ids=[0], dest_expansion_ids=[1], src_expansion_positions=[3], dest_expansion_positions=[3],
        value=torch.tensor([[6.]]), replicate_levels=[False],
    )

    force_expected = torch.tensor([[3.]] * 3 + [[6.]] + [[-1.]] * 4 + [[3.]] * 3 + [[6.]] + [[-1.]] * 4)

    assert torch.equal(data.force, force_expected)


@pytest.mark.parametrize("method", ["set_edge_attr_per_expansion", "set_edge_attr_per_expansion_by_pattern"])
def test_set_edge_attr_per_expansion_invalid_input(method):

    data = build_structure()

    # expansion 0, position 0 (node 0) and expansion 1, position 2 (node 6) are not connected
    with pytest.raises(ValueError, match="No edge found"):
        getattr(data, method)("force", [0], [1], [0], [2], torch.tensor([[1.]]))

    # expansion 3 does not exist
    with pytest.raises(ValueError, match="No node found"):
        getattr(data, method)("force", [3], [1], [0], [0], torch.tensor([[1.]]))

    # a pattern is registered, but no node belongs to an expansion
    data = build_structure(nodes_by_pattern=False)
    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [1., 0., 0.]]))
    data.add_edges(edge_indices=torch.tensor([[0], [1]]), force=torch.tensor([[1.]]))

    with pytest.raises(ValueError, match="No node found"):
        getattr(data, method)("force", [0], [0], [0], [1], torch.tensor([[1.]]))

    # the nodes exist, but the graph has no edges at all
    data = build_structure(nodes_by_pattern=False)
    data.add_nodes_by_pattern(pattern="rot4", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))

    with pytest.raises(ValueError, match="No edge found"):
        getattr(data, method)("force", [0], [1], [0], [0], torch.tensor([[1.]]))


def test_set_edge_attr_per_expansion_by_pattern_refuses_a_merged_end_that_changes_the_edge():

    data = StructData(
        node_attrs={"coords": torch.empty((0, 3), dtype=torch.float)},
        edge_attrs={"force": torch.empty((0, 1), dtype=torch.float)},
    )
    x = torch.tensor([1., 0., 0.])
    data.add_patterns({"s": data.nest_patterns([data.create_translational_pattern(4, x), data.create_mirror_pattern(normal=x)])})

    # the array starts on the mirror plane, so position 4, its mirror image, is merged into position 0
    data.add_nodes_by_pattern(pattern="s", coords=torch.tensor([[0., 0., 0.]]))
    data.add_edges_per_expansion_by_pattern(
        src_expansion_ids=[0], dest_expansion_ids=[0], src_expansion_positions=[0], dest_expansion_positions=[1],
        force=torch.tensor([[1.]]),
    )

    # the edge from position 4 to 5 exists, but from position 0 its copies would cross the mirror plane
    with pytest.raises(ValueError, match="an end of it was merged into another copy"):
        data.set_edge_attr_per_expansion_by_pattern("force", [0], [0], [4], [5], torch.tensor([[2.]]))

    assert torch.equal(data.force, torch.ones_like(data.force))

    # given where nothing was merged, the whole chain is set
    data.set_edge_attr_per_expansion_by_pattern("force", [0], [0], [0], [1], torch.tensor([[2.]]))

    assert torch.equal(data.force, torch.full_like(data.force, 2.))


@pytest.mark.parametrize("method", ["set_edge_attr_per_expansion", "set_edge_attr_per_expansion_by_pattern"])
@pytest.mark.parametrize("num_rows", [0, 2])
def test_set_edge_attr_per_expansion_needs_one_value_row_per_entry(method, num_rows):

    data = build_structure()
    force_before = data.force.clone()

    # one edge, the radial edge 0 -> 4, but no or two rows of value
    with pytest.raises(ValueError, match=rf"value must have one row per entry \(1\), but has {num_rows}"):
        getattr(data, method)("force", [0], [1], [0], [0], torch.ones((num_rows, 1)))

    assert torch.equal(data.force, force_before)
