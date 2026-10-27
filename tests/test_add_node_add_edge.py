import pytest
import torch
from torch_structure.data.data import StructData


def build_structure():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "is_support": torch.empty((0, 1), dtype=torch.bool),
    }
    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }
    default_attrs = {"force": torch.tensor([-1.0])}

    return StructData(node_attrs=node_attrs, edge_attrs=edge_attrs, default_attrs=default_attrs)


def test_add_node():

    data = build_structure()

    data.add_node("A", coords=[0., 0., 1.], is_support=True)
    data.add_node("B", coords=torch.tensor([1., 0., 0.]))

    coords_expected = torch.tensor([[0., 0., 1.], [1., 0., 0.]])
    is_support_expected = torch.tensor([[True], [False]])

    assert data.num_nodes == 2
    assert torch.equal(data.coords, coords_expected)
    assert torch.equal(data.is_support, is_support_expected)
    assert data.name.shape == (2,)
    assert torch.equal(data.get_node_index_from_name("B"), torch.tensor([1]))


def test_add_node_without_attributes():

    data = build_structure()

    data.add_node("A")

    assert data.num_nodes == 1
    assert torch.equal(data.coords, torch.zeros((1, 3)))
    assert torch.equal(data.get_node_index_from_name("A"), torch.tensor([0]))


def test_add_node_casts_to_registered_dtype():

    data = build_structure()

    data.add_node("A", coords=torch.tensor([0., 0., 1.], dtype=torch.float64))

    assert data.coords.dtype == torch.float


def test_add_node_with_integer_name():

    data = build_structure()

    data.add_node(3, coords=[0., 0., 0.])

    assert list(data.nodes) == ["3"]
    assert torch.equal(data.get_node_index_from_name(3), torch.tensor([0]))


def test_add_node_mixed_with_add_nodes():

    data = build_structure()

    data.add_node("A", coords=[0., 0., 0.])
    data.add_nodes(names=["B", "C"], coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]))
    data.add_node("D", coords=[3., 0., 0.])

    assert data.name.shape == (4,)
    assert list(data.nodes) == ["A", "B", "C", "D"]

    # names from either method are found by add_edge
    data.add_edge("A", "C")
    assert torch.equal(data.edge_index, torch.tensor([[0, 2], [2, 0]]))


def test_add_node_invalid_input():

    data = build_structure()
    data.add_node("A", coords=[0., 0., 0.])

    with pytest.raises(ValueError, match="already exists"):
        data.add_node("A", coords=[1., 0., 0.])

    with pytest.raises(ValueError, match="exceeds the 8-byte limit"):
        data.add_node("too_long_name", coords=[1., 0., 0.])

    with pytest.raises(ValueError, match="Unexpected node attributes"):
        data.add_node("B", cords=[1., 0., 0.])


def test_add_edge():

    data = build_structure()
    data.add_nodes(names=["A", "B", "C"], coords=torch.zeros((3, 3)))

    data.add_edge("A", "B", force=2.0)
    data.add_edge("C", "B")

    # every edge is stored as a directed edge followed by its reciprocal
    edge_index_expected = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]])
    directed_mask_expected = torch.tensor([[True], [False], [True], [False]])
    reciprocal_edge_expected = torch.tensor([[1], [0], [3], [2]])
    force_expected = torch.tensor([[2.], [2.], [-1.], [-1.]])

    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.directed_mask, directed_mask_expected)
    assert torch.equal(data.reciprocal_edge, reciprocal_edge_expected)
    assert torch.equal(data.force, force_expected)


def test_add_edge_invalid_input():

    data = build_structure()
    data.add_nodes(names=["A", "B", "C"], coords=torch.zeros((3, 3)))
    data.add_edge("A", "B")

    with pytest.raises(ValueError, match="No node found with name 'Z'"):
        data.add_edge("A", "Z")

    with pytest.raises(ValueError, match="already exist"):
        data.add_edge("B", "A")

    with pytest.raises(ValueError, match="Unexpected edge attributes"):
        data.add_edge("A", "C", forse=1.0)


def test_add_edge_with_ambiguous_name():

    data = build_structure()
    data.add_nodes(names=["A", "A", "B"], coords=torch.zeros((3, 3)))

    with pytest.raises(ValueError, match="ambiguous"):
        data.add_edge("A", "B")


def test_nodes_view():

    data = build_structure()
    data.add_nodes(names=["A", "B"], coords=torch.tensor([[0., 0., 0.], [1., 2., 3.]]))
    data.add_nodes(coords=torch.tensor([[5., 5., 5.]]))  # unnamed

    # unnamed nodes are left out of the view
    assert list(data.nodes) == ["A", "B"]
    assert len(data.nodes) == 2
    assert data.nodes.index("B") == 1
    assert torch.equal(data.nodes["B"]["coords"], torch.tensor([1., 2., 3.]))

    data.nodes["A"]["coords"] = torch.tensor([9., 9., 9.])
    assert torch.equal(data.coords[0], torch.tensor([9., 9., 9.]))

    with pytest.raises(KeyError, match="No node found with name 'Z'"):
        data.nodes["Z"]


def test_merge_nodes():

    data = build_structure()

    #   B       D
    #    \     /
    #     A   C        merge C into A
    data.add_nodes(
        names=["A", "B", "C", "D"],
        coords=torch.tensor([[0., 0., 0.], [-1., 1., 0.], [1., 0., 0.], [2., 1., 0.]]),
    )
    data.add_edge("A", "B")
    data.add_edge("C", "D")

    data.merge_nodes("A", ["C"])

    edge_index_expected = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]])
    coords_expected = torch.tensor([[0., 0., 0.], [-1., 1., 0.], [2., 1., 0.]])

    assert data.num_nodes == 3
    assert list(data.nodes) == ["A", "B", "D"]
    assert torch.equal(data.edge_index, edge_index_expected)
    assert torch.equal(data.coords, coords_expected)

    with pytest.raises(ValueError, match="No node found with name 'C'"):
        data.merge_nodes("A", ["C"])
