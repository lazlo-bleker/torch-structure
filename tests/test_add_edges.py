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
    data.add_nodes(symmetry="rot4", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))

    # a radial edge 0 -> 4 and a ring edge 0 -> 1, each copied to all 4 orbit positions
    edge_indices = torch.tensor([[0, 0], [4, 1]])

    new_edge_attrs = {
        "force": torch.tensor([[3.], [-1.]]),
    }

    data.add_edges(edge_indices=edge_indices, **new_edge_attrs)

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
    data.add_nodes(symmetry="d4", coords=torch.tensor([[2., 1., 0.]]))

    # 0 -> 1 differs on the mirror level, so it is copied on both levels; its 8 copies contain every
    # edge twice, which leaves 4 edges. 0 -> 3 also differs on the rotation level, so it is only
    # copied on the rotation level.
    edge_indices = torch.tensor([[0, 0], [1, 3]])

    new_edge_attrs = {
        "force": torch.tensor([[1.], [2.]]),
    }

    data.add_edges(edge_indices=edge_indices, **new_edge_attrs)

    # each edge family keeps its own force
    edge_index_expected = torch.tensor([
        [0, 2, 4, 6, 0, 2, 4, 6, 1, 3, 5, 7, 3, 5, 7, 1],
        [1, 3, 5, 7, 3, 5, 7, 1, 0, 2, 4, 6, 0, 2, 4, 6],
    ])
    force_expected = torch.tensor([[1.]] * 4 + [[2.]] * 4 + [[1.]] * 4 + [[2.]] * 4)

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
    data.add_nodes(symmetry="rot4", coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))
    data.add_nodes(coords=torch.tensor([[0., 0., 3.]]))
    data.add_nodes(symmetry="mirror", coords=torch.tensor([[3., 1., 0.]]))

    # with consider_symmetry=False only the given edge is added
    data.add_edges(edge_indices=torch.tensor([[1], [6]]), consider_symmetry=False, force=torch.tensor([[5.]]))

    # an edge to a node outside any symmetry is added as given
    data.add_edges(edge_indices=torch.tensor([[8], [2]]), force=torch.tensor([[1.]]))

    edge_index_expected = torch.tensor([[1, 6, 8, 2], [6, 1, 2, 8]])
    force_expected = torch.tensor([[5.], [5.], [1.], [1.]])

    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.force, force_expected)

    # an edge between two different symmetries cannot be copied
    with pytest.raises(ValueError, match="two different registered symmetries"):
        data.add_edges(edge_indices=torch.tensor([[0], [9]]), force=torch.tensor([[1.]]))
   


