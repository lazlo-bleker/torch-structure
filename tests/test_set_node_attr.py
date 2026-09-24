import pytest
import torch
from torch_structure.data.data import StructData


def build_structure():

    # stiffness is neither a transform nor a copy attribute of the symmetry
    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
        "stiffness": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    data.add_symmetry({"rot4": data.create_rotational_symmetry(4)}, copy_attrs=["load"])

    # orbit 0 (nodes 0-3) and orbit 1 (nodes 4-7)
    data.add_nodes(
        symmetry="rot4",
        coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]),
        load=torch.tensor([[0., 0., -1.], [0., 0., -2.]]),
    )

    return data


def test_set_node_attr_transform_attribute():

    data = build_structure()

    # select node 2 (orbit 0, position 2): the seed is solved from its new value and the orbit is rebuilt
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[2] = True

    data.set_node_attr("coords", mask, torch.tensor([[-2., 0., 3.]]))

    coords_expected = torch.tensor([
        [2., 0., 3.], [0., 2., 3.], [-2., 0., 3.], [0., -2., 3.],
        [2., 0., 0.], [0., 2., 0.], [-2., 0., 0.], [0., -2., 0.],
    ])

    assert torch.allclose(data.coords, coords_expected, atol=1e-6)


def test_set_node_attr_copy_attribute_and_single_nodes():

    data = build_structure()

    # node 8 is outside any symmetry
    data.add_nodes(coords=torch.tensor([[0., 0., 3.]]))

    # a copy attribute set via node 5 (orbit 1, position 1) is copied to the whole orbit
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[5] = True
    data.set_node_attr("load", mask, torch.tensor([[1., 2., 3.]]))

    # with consider_symmetry=False only node 3 changes
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[3] = True
    data.set_node_attr("load", mask, torch.tensor([[0., 0., -5.]]), consider_symmetry=False)

    # a node outside any symmetry only changes itself
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[8] = True
    data.set_node_attr("load", mask, torch.tensor([[0., 0., -7.]]))

    load_expected = torch.tensor([
        [0., 0., -1.], [0., 0., -1.], [0., 0., -1.], [0., 0., -5.],
        [1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.],
        [0., 0., -7.],
    ])

    assert torch.equal(data.load, load_expected)


def test_set_node_attr_invalid_input():

    data = build_structure()

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[0] = True

    with pytest.raises(ValueError, match="symmetry bookkeeping"):
        data.set_node_attr("orbit_id", mask, torch.tensor([[1]]))

    with pytest.raises(ValueError, match="mask must have 8 entries"):
        data.set_node_attr("load", torch.zeros(5, dtype=torch.bool), torch.zeros((0, 3)))

    with pytest.raises(ValueError, match="one row per masked node"):
        data.set_node_attr("load", mask, torch.zeros((2, 3)))

    # nodes 0 and 1 belong to the same orbit, which is not allowed for any attribute, even with
    # the same value or with values that are symmetry copies
    two_nodes = torch.zeros(data.num_nodes, dtype=torch.bool)
    two_nodes[[0, 1]] = True

    with pytest.raises(ValueError, match="same orbit"):
        data.set_node_attr("load", two_nodes, torch.tensor([[0., 0., -5.], [0., 0., -5.]]))

    with pytest.raises(ValueError, match="same orbit"):
        data.set_node_attr("coords", two_nodes, torch.tensor([[2., 0., 3.], [0., 2., 3.]]))

    with pytest.raises(ValueError, match="not classified as a transform or copy attribute"):
        data.set_node_attr("stiffness", mask, torch.tensor([[4.]]))
