import pytest
import torch
from torch_structure.data.data import StructData


def build_structure():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    data.add_symmetry({"rot4": data.create_rotational_symmetry(4)}, copy_attrs=["load"])

    # orbit 0 named "A" (nodes 0-3), orbit 1 named "B" (nodes 4-7), and node 8, also named "A" but
    # outside any symmetry
    data.add_nodes(
        names=["A", "B"],
        symmetry="rot4",
        coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]),
        load=torch.tensor([[0., 0., -1.], [0., 0., -2.]]),
    )
    data.add_nodes(names=["A"], coords=torch.tensor([[9., 9., 9.]]))

    return data


def test_set_node_attr_by_name_with_symmetry():

    data = build_structure()

    # a copy attribute: "B" matches orbit 1, "A" matches orbit 0 and node 8
    data.set_node_attr_by_name("load", ["B", "A"], torch.tensor([[0., 0., -6.], [0., 0., -7.]]))

    # a transform attribute needs a name with only one node per orbit: after renaming node 6
    # (orbit 1, position 2), "B2" matches only that node, and orbit 1 is rebuilt from position 2
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[6] = True
    data.name_nodes(["B2"], mask)

    data.set_node_attr_by_name("coords", ["B2"], torch.tensor([[0., -3., 1.]]))

    load_expected = torch.tensor([[0., 0., -7.]] * 4 + [[0., 0., -6.]] * 4 + [[0., 0., -7.]])
    coords_expected = torch.tensor([
        [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.],
        [0., 3., 1.], [-3., 0., 1.], [0., -3., 1.], [3., 0., 1.],
        [9., 9., 9.],
    ])

    assert torch.equal(data.load, load_expected)
    assert torch.allclose(data.coords, coords_expected, atol=1e-6)


def test_set_node_attr_by_name_without_symmetry():

    data = build_structure()

    data.set_node_attr_by_name("coords", ["A"], torch.tensor([[0., 5., 0.]]), consider_symmetry=False)

    # every node named "A" gets the value as given, so the orbit collapses onto one point; orbit 1 is unchanged
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

    # "A" matches all nodes of orbit 0, which is not allowed for a transform attribute
    with pytest.raises(ValueError, match="several nodes of the same orbit"):
        data.set_node_attr_by_name("coords", ["A"], torch.zeros((1, 3)))
