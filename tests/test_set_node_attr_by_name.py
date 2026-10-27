import pytest
import torch
from torch_structure.data.data import StructData


def build_structure():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    data.add_patterns({"rot4": data.create_rotational_pattern(4)})

    # expansion 0 named "A" (nodes 0-3), expansion 1 named "B" (nodes 4-7), and node 8, also named "A" but
    # outside any pattern
    data.add_nodes_by_pattern(
        names=["A", "B"],
        pattern="rot4",
        coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]),
        load=torch.tensor([[0., 0., -1.], [0., 0., -2.]]),
    )
    data.add_nodes(names=["A"], coords=torch.tensor([[9., 9., 9.]]))

    return data


def test_set_node_attr_by_name_without_pattern():

    data = build_structure()

    data.set_node_attr_by_name("coords", ["A"], torch.tensor([[0., 5., 0.]]))

    # every node named "A" gets the value as given, so the expansion collapses onto one point; expansion 1 is unchanged
    coords_expected = torch.tensor([
        [0., 5., 0.], [0., 5., 0.], [0., 5., 0.], [0., 5., 0.],
        [2., 0., 0.], [0., 2., 0.], [-2., 0., 0.], [0., -2., 0.],
        [0., 5., 0.],
    ])

    assert torch.allclose(data.coords, coords_expected, atol=1e-6)


def test_set_node_attr_by_name_invalid_input():

    data = build_structure()

    with pytest.raises(ValueError, match="must not contain duplicates"):
        data.set_node_attr_by_name("load", ["A", "A"], torch.zeros((2, 3)))

    with pytest.raises(ValueError, match="No node found with name 'Z'"):
        data.set_node_attr_by_name("load", ["Z"], torch.zeros((1, 3)))

    with pytest.raises(ValueError, match="one row per name"):
        data.set_node_attr_by_name("load", ["A", "B"], torch.zeros((1, 3)))


def test_set_node_attr_by_name_on_a_graph_without_pattern():

    node_attrs = {
        "load": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    data.add_nodes(names=["A"], load=torch.zeros((1, 3)))

    # set_node_attr_by_name does not need a pattern
    data.set_node_attr_by_name("load", ["A"], torch.ones((1, 3)))

    assert torch.equal(data.load, torch.ones((1, 3)))
