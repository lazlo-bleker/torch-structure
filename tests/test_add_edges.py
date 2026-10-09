import pytest
import torch
from torch_structure.data.data import StructData

def test_add_edges():

    edge_attrs = {
        "force_densities": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(edge_attrs=edge_attrs)

    fd = torch.tensor([[0], [3], [2]])

    new_edge_attrs = {
        "force_densities": fd,
    }

    data.add_n_empty_nodes(3)

    edge_indices = torch.tensor([[0,1,2], [1,2,0]])
    data.add_edges(edge_indices=edge_indices, **new_edge_attrs)


def test_add_edges_with_rotational_symmetry():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    data.add_symmetry({"rot4": data.create_rotational_symmetry(4)})

    # orbit 0 (nodes 0-3) and orbit 1 (nodes 4-7)
    data.add_nodes_symmetrical(symmetry="rot4", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))

    # a radial edge 0 -> 4 and a ring edge 0 -> 1, each copied to all 4 orbit positions
    edge_indices = torch.tensor([[0, 0], [4, 1]])

    new_edge_attrs = {
        "force": torch.tensor([[3.], [-1.]]),
    }

    data.add_edges_symmetrical(edge_indices=edge_indices, **new_edge_attrs)

    edge_index_expected = torch.tensor([
        [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 0],
        [4, 5, 6, 7, 1, 2, 3, 0, 0, 1, 2, 3, 0, 1, 2, 3],
    ])
    force_expected = torch.tensor([[3.]] * 4 + [[-1.]] * 4 + [[3.]] * 4 + [[-1.]] * 4)
    directed_mask_expected = torch.tensor([True] * 8 + [False] * 8).unsqueeze(-1)
    reciprocal_edge_expected = torch.tensor([8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]).unsqueeze(-1)

    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.force, force_expected)
    assert torch.equal(data.directed_mask, directed_mask_expected)
    assert torch.equal(data.reciprocal_edge, reciprocal_edge_expected)


def test_add_edges_with_combined_symmetry():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    mirror = data.create_mirror_symmetry(normal=torch.tensor([0., 1., 0.]))
    rotation = data.create_rotational_symmetry(4)
    data.add_symmetry({"d4": data.combine_symmetry(mirror, rotation)})

    # one orbit of 8 nodes: orbit position p is the seed, mirrored if p is odd, then rotated by 90° * (p // 2)
    data.add_nodes_symmetrical(symmetry="d4", coords=torch.tensor([[2., 1., 0.]]))

    # every edge is copied over the full symmetry. 0 -> 1 differs on the mirror level only, so its
    # 8 copies contain every edge twice, which leaves 4 edges. 0 -> 3 differs on both levels, and
    # its 8 copies are all distinct.
    edge_indices = torch.tensor([[0, 0], [1, 3]])

    new_edge_attrs = {
        "force": torch.tensor([[1.], [2.]]),
    }

    data.add_edges_symmetrical(edge_indices=edge_indices, **new_edge_attrs)

    # each edge family keeps its own force
    edge_index_expected = torch.tensor([
        [0, 2, 4, 6, 0, 1, 2, 3, 4, 5, 6, 7, 1, 3, 5, 7, 3, 2, 5, 4, 7, 6, 1, 0],
        [1, 3, 5, 7, 3, 2, 5, 4, 7, 6, 1, 0, 0, 2, 4, 6, 0, 1, 2, 3, 4, 5, 6, 7],
    ])
    force_expected = torch.tensor([[1.]] * 4 + [[2.]] * 8 + [[1.]] * 4 + [[2.]] * 8)

    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.force, force_expected)


def test_add_edges_symmetry_exceptions():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    data.add_symmetry({"rot4": data.create_rotational_symmetry(4), "mirror": data.create_mirror_symmetry()})

    # nodes 0-7: two rot4 orbits, node 8: outside any symmetry, nodes 9-10: a mirror orbit
    data.add_nodes_symmetrical(symmetry="rot4", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))
    data.add_nodes(coords=torch.tensor([[0., 0., 3.]]))
    data.add_nodes_symmetrical(symmetry="mirror", coords=torch.tensor([[3., 1., 0.]]))

    # add_edges only adds the given edge, even between nodes of a symmetry
    data.add_edges(edge_indices=torch.tensor([[1], [6]]), force=torch.tensor([[5.]]))

    # an edge to a node outside any symmetry is added via add_edges
    data.add_edges(edge_indices=torch.tensor([[8], [2]]), force=torch.tensor([[1.]]))

    edge_index_expected = torch.tensor([[1, 6, 8, 2], [6, 1, 2, 8]])
    force_expected = torch.tensor([[5.], [5.], [1.], [1.]])

    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.force, force_expected)

    # an edge to a node outside any symmetry cannot be copied, not even together with one that can
    with pytest.raises(ValueError, match="outside any symmetry"):
        data.add_edges_symmetrical(edge_indices=torch.tensor([[0, 8], [1, 3]]), force=torch.tensor([[1.], [1.]]))

    # an edge between two different symmetries cannot be copied
    with pytest.raises(ValueError, match="two different registered symmetries"):
        data.add_edges_symmetrical(edge_indices=torch.tensor([[0], [9]]), force=torch.tensor([[1.]]))

    # edges of two different symmetries cannot be added in the same call
    with pytest.raises(ValueError, match="several registered symmetries"):
        data.add_edges_symmetrical(edge_indices=torch.tensor([[0, 9], [1, 10]]), force=torch.tensor([[1.], [1.]]))

    # the failed calls left the graph unchanged
    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.force, force_expected)


def test_add_edges_rejects_repeated_and_existing_edges():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    data.add_symmetry({"rot4": data.create_rotational_symmetry(4)})

    # orbit 0 (nodes 0-3) and orbit 1 (nodes 4-7)
    data.add_nodes_symmetrical(symmetry="rot4", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))

    # the same edge in both directions
    with pytest.raises(ValueError, match="given twice"):
        data.add_edges(edge_indices=torch.tensor([[0, 4], [4, 0]]), force=torch.tensor([[1.], [1.]]))

    data.add_edges(edge_indices=torch.tensor([[0], [4]]), force=torch.tensor([[1.]]))

    with pytest.raises(ValueError, match="already exist"):
        data.add_edges(edge_indices=torch.tensor([[4], [0]]), force=torch.tensor([[1.]]))

    # two radial edges are symmetry copies of each other
    with pytest.raises(ValueError, match="symmetry copies of each other"):
        data.add_edges_symmetrical(edge_indices=torch.tensor([[1, 2], [5, 6]]), force=torch.tensor([[1.], [1.]]))

    # the copies of the radial edge 1 -> 5 include the existing edge 0 -> 4
    with pytest.raises(ValueError, match="already exist"):
        data.add_edges_symmetrical(edge_indices=torch.tensor([[1], [5]]), force=torch.tensor([[1.]]))

    # the failed calls left the graph unchanged
    assert torch.equal(data.edge_index, torch.tensor([[0, 4], [4, 0]]))


def test_add_edges_symmetrical_without_symmetry():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [1., 0., 0.]]))

    with pytest.raises(ValueError, match="No symmetry is registered"):
        data.add_edges_symmetrical(edge_indices=torch.tensor([[0], [1]]), force=torch.tensor([[1.]]))

    # add_edges does not need a symmetry
    data.add_edges(edge_indices=torch.tensor([[0], [1]]), force=torch.tensor([[1.]]))

    assert torch.equal(data.edge_index, torch.tensor([[0, 1], [1, 0]]))


def test_add_edges_rejects_attributes_without_one_row_per_edge():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

    data.add_symmetry({"rot4": data.create_rotational_symmetry(4)}, copy_attrs=["force"])

    # orbit 0 (nodes 0-3) and orbit 1 (nodes 4-7)
    data.add_nodes_symmetrical(symmetry="rot4", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))

    for add_edges in (data.add_edges, data.add_edges_symmetrical):

        # one edge, two rows
        with pytest.raises(ValueError, match="one row per edge"):
            add_edges(edge_indices=torch.tensor([[0], [4]]), force=torch.tensor([[1.], [2.]]))

        # two edges, one row
        with pytest.raises(ValueError, match="one row per edge"):
            add_edges(edge_indices=torch.tensor([[0, 0], [4, 1]]), force=torch.tensor([[1.]]))

    # the failed calls left the graph unchanged
    assert data.edge_index.shape[1] == 0
    assert data.force.shape[0] == 0

    # values that are not tensors are counted the same way
    data.add_edges(edge_indices=torch.tensor([[0], [4]]), force=[[2.]])

    with pytest.raises(ValueError, match="one row per edge"):
        data.add_edges(edge_indices=torch.tensor([[1], [5]]), force=[[2.], [3.]])

    assert torch.equal(data.edge_index, torch.tensor([[0, 4], [4, 0]]))
    assert torch.equal(data.force, torch.tensor([[2.], [2.]]))
