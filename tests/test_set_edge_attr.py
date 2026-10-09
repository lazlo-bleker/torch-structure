import pytest
import torch
from torch_structure.data.data import StructData


def build_rot4_structure():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    # stiffness is not a copy attribute of the symmetry
    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
        "stiffness": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    data.add_symmetry({"rot4": data.create_rotational_symmetry(4)}, copy_attrs=["force"])

    # orbit 0 (nodes 0-3) and orbit 1 (nodes 4-7)
    data.add_nodes_symmetrical(symmetry="rot4", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))

    # radial edges: rows 0-3 (0 -> 4, 1 -> 5, 2 -> 6, 3 -> 7), ring edges: rows 4-7 (0 -> 1, 1 -> 2, 2 -> 3, 3 -> 0),
    # reciprocal rows 8-15
    data.add_edges_symmetrical(edge_indices=torch.tensor([[0, 0], [4, 1]]), force=torch.tensor([[3.], [-1.]]))

    return data


def test_set_edge_attr_propagates_to_edge_family():

    data = build_rot4_structure()

    # select the radial edge 2 -> 6 by its reciprocal row (row 10: 6 -> 2)
    mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
    mask[10] = True
    data.set_edge_attr_symmetrical("force", mask, torch.tensor([[7.]]))

    # set_edge_attr only changes the ring edge 1 -> 2 (row 5) and its reciprocal row
    mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
    mask[5] = True
    data.set_edge_attr("force", mask, torch.tensor([[4.]]))

    force_expected = torch.tensor(
        [[7.]] * 4 + [[-1.], [4.], [-1.], [-1.]] + [[7.]] * 4 + [[-1.], [4.], [-1.], [-1.]]
    )

    assert torch.equal(data.force, force_expected)


def test_set_edge_attr_with_combined_symmetry():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    mirror = data.create_mirror_symmetry(normal=torch.tensor([0., 1., 0.]))
    rotation = data.create_rotational_symmetry(4)
    data.add_symmetry({"d4": data.combine_symmetry(mirror, rotation)}, copy_attrs=["force"])

    # nodes 0-7: one orbit, node 8: outside any symmetry
    data.add_nodes_symmetrical(symmetry="d4", coords=torch.tensor([[2., 1., 0.]]))
    data.add_nodes(coords=torch.tensor([[0., 0., 3.]]))

    # the edges to node 8 are added as given: rows 0-1, reciprocal rows 2-3
    data.add_edges(edge_indices=torch.tensor([[8, 8], [0, 1]]), force=torch.zeros((2, 1)))

    # the mirror family (rows 4-7) and the diagonal family (rows 8-15); rows 16-27 are the reciprocal rows
    data.add_edges_symmetrical(edge_indices=torch.tensor([[0, 0], [1, 3]]), force=torch.zeros((2, 1)))

    edge_index_forward_expected = torch.tensor([
        [8, 8, 0, 2, 4, 6, 0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 1, 3, 5, 7, 3, 2, 5, 4, 7, 6, 1, 0],
    ])

    assert torch.equal(data.edge_index[:, data.directed_mask.view(-1)], edge_index_forward_expected)

    # one edge of each family in the same call: row 6 (4 -> 5) and row 12 (4 -> 7)
    mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
    mask[[6, 12]] = True
    data.set_edge_attr_symmetrical("force", mask, torch.tensor([[2.], [9.]]))

    # an edge to node 8 is set via set_edge_attr
    mask = torch.zeros(data.edge_index.shape[1], dtype=torch.bool)
    mask[0] = True
    data.set_edge_attr("force", mask, torch.tensor([[5.]]))

    force_expected = torch.tensor(
        [[5.], [0.], [5.], [0.]] + [[2.]] * 4 + [[9.]] * 8 + [[2.]] * 4 + [[9.]] * 8
    )

    assert torch.equal(data.force, force_expected)


def test_set_edge_attr_invalid_input():

    data = build_rot4_structure()

    num_rows = data.edge_index.shape[1]

    # the forward row 0 (0 -> 4) and its reciprocal row 8 (4 -> 0)
    both_rows = torch.zeros(num_rows, dtype=torch.bool)
    both_rows[[0, 8]] = True

    mask = torch.zeros(num_rows, dtype=torch.bool)
    mask[0] = True

    # the checks both versions share
    for set_edge_attr in (data.set_edge_attr, data.set_edge_attr_symmetrical):

        with pytest.raises(ValueError, match="both the forward and reciprocal row"):
            set_edge_attr("force", both_rows, torch.tensor([[1.], [2.]]))

        with pytest.raises(ValueError, match="one row per masked edge"):
            set_edge_attr("force", mask, torch.tensor([[1.], [2.]]))

        with pytest.raises(ValueError, match=f"mask must have {num_rows} entries"):
            set_edge_attr("force", torch.zeros(num_rows - 1, dtype=torch.bool), torch.zeros((0, 1)))

    with pytest.raises(ValueError, match="not registered as a copy attribute"):
        data.set_edge_attr_symmetrical("stiffness", mask, torch.tensor([[1.]]))

    # rot4 has a single level
    with pytest.raises(ValueError, match="one entry per symmetry level"):
        data.set_edge_attr_symmetrical("force", mask, torch.tensor([[1.]]), replicate_levels=[True, False])

    # two radial edges, i.e. symmetry copies of each other, with different values
    mask = torch.zeros(num_rows, dtype=torch.bool)
    mask[[0, 2]] = True

    with pytest.raises(ValueError, match="symmetry copies of each other"):
        data.set_edge_attr_symmetrical("force", mask, torch.tensor([[1.], [2.]]))

    # with the same value for both there is no conflict
    data.set_edge_attr_symmetrical("force", mask, torch.tensor([[1.], [1.]]))

    assert torch.equal(data.force[:4], torch.ones((4, 1)))

    # set_edge_attr does not need a copy attribute
    mask = torch.zeros(num_rows, dtype=torch.bool)
    mask[0] = True
    data.set_edge_attr("stiffness", mask, torch.tensor([[2.]]))

    assert torch.equal(data.stiffness[[0, 8]], torch.tensor([[2.], [2.]]))


def test_set_edge_attr_symmetrical_rejects_edges_outside_one_symmetry():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    data.add_symmetry(
        {"rot4": data.create_rotational_symmetry(4), "mirror": data.create_mirror_symmetry()}, copy_attrs=["force"]
    )

    # nodes 0-3: a rot4 orbit, nodes 4-5: a mirror orbit, node 6: outside any symmetry
    data.add_nodes_symmetrical(symmetry="rot4", coords=torch.tensor([[1., 0., 0.]]))
    data.add_nodes_symmetrical(symmetry="mirror", coords=torch.tensor([[2., 1., 0.]]))
    data.add_nodes(coords=torch.tensor([[0., 0., 3.]]))

    # the rot4 ring: rows 0-3, reciprocal rows 4-7
    data.add_edges_symmetrical(edge_indices=torch.tensor([[0], [1]]), force=torch.zeros((1, 1)))

    # the edge across the mirror: row 8, reciprocal row 9
    data.add_edges_symmetrical(edge_indices=torch.tensor([[4], [5]]), force=torch.zeros((1, 1)))

    # 0 -> 4 between the two symmetries and 0 -> 6 to node 6: rows 10-11, reciprocal rows 12-13
    data.add_edges(edge_indices=torch.tensor([[0, 0], [4, 6]]), force=torch.zeros((2, 1)))

    num_rows = data.edge_index.shape[1]

    def mask_of(rows):
        mask = torch.zeros(num_rows, dtype=torch.bool)
        mask[rows] = True
        return mask

    with pytest.raises(ValueError, match="outside any symmetry"):
        data.set_edge_attr_symmetrical("force", mask_of([11]), torch.tensor([[1.]]))

    with pytest.raises(ValueError, match="two different registered symmetries"):
        data.set_edge_attr_symmetrical("force", mask_of([10]), torch.tensor([[1.]]))

    # edges of two different symmetries cannot be set in the same call
    with pytest.raises(ValueError, match="several registered symmetries"):
        data.set_edge_attr_symmetrical("force", mask_of([0, 8]), torch.tensor([[1.], [2.]]))

    # the failed calls left the attribute unchanged
    assert torch.equal(data.force, torch.zeros((num_rows, 1)))

    # set_edge_attr sets any edge, and set_edge_attr_symmetrical works with one call per symmetry
    data.set_edge_attr("force", mask_of([10, 11]), torch.tensor([[3.], [4.]]))
    data.set_edge_attr_symmetrical("force", mask_of([0]), torch.tensor([[1.]]))
    data.set_edge_attr_symmetrical("force", mask_of([8]), torch.tensor([[2.]]))

    force_expected = torch.tensor([[1.]] * 8 + [[2.]] * 2 + [[3.], [4.], [3.], [4.]])

    assert torch.equal(data.force, force_expected)


def test_set_edge_attr_symmetrical_without_symmetry():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [1., 0., 0.]]))
    data.add_edges(edge_indices=torch.tensor([[0], [1]]), force=torch.zeros((1, 1)))

    mask = torch.tensor([True, False])

    with pytest.raises(ValueError, match="No symmetry is registered"):
        data.set_edge_attr_symmetrical("force", mask, torch.tensor([[1.]]))

    # set_edge_attr does not need a symmetry
    data.set_edge_attr("force", mask, torch.tensor([[1.]]))

    assert torch.equal(data.force, torch.ones((2, 1)))
