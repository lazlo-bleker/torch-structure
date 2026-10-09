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
    data.add_nodes_symmetrical(
        symmetry="rot4",
        coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]),
        load=torch.tensor([[0., 0., -1.], [0., 0., -2.]]),
    )

    return data


def test_set_node_attr_symmetrical_transform_attribute():

    data = build_structure()

    # select node 2 (orbit 0, position 2): the seed is solved from its new value and the orbit is rebuilt
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[2] = True

    data.set_node_attr_symmetrical("coords", mask, torch.tensor([[-2., 0., 3.]]))

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
    data.set_node_attr_symmetrical("load", mask, torch.tensor([[1., 2., 3.]]))

    # set_node_attr only changes node 3
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[3] = True
    data.set_node_attr("load", mask, torch.tensor([[0., 0., -5.]]))

    # a node outside any symmetry is set via set_node_attr
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[8] = True
    data.set_node_attr("load", mask, torch.tensor([[0., 0., -7.]]))

    load_expected = torch.tensor([
        [0., 0., -1.], [0., 0., -1.], [0., 0., -1.], [0., 0., -5.],
        [1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.],
        [0., 0., -7.],
    ])

    assert torch.equal(data.load, load_expected)

    # set_node_attr_symmetrical rejects a node outside any symmetry, even together with one it can
    # propagate from, and leaves the attribute unchanged
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[[5, 8]] = True

    with pytest.raises(ValueError, match="outside any symmetry"):
        data.set_node_attr_symmetrical("load", mask, torch.tensor([[9., 9., 9.], [9., 9., 9.]]))

    assert torch.equal(data.load, load_expected)


def test_set_node_attr_invalid_input():

    data = build_structure()

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[0] = True

    # the checks both versions share
    for set_node_attr in (data.set_node_attr, data.set_node_attr_symmetrical):

        with pytest.raises(ValueError, match="symmetry bookkeeping"):
            set_node_attr("orbit_id", mask, torch.tensor([[1]]))

        with pytest.raises(ValueError, match="mask must have 8 entries"):
            set_node_attr("load", torch.zeros(5, dtype=torch.bool), torch.zeros((0, 3)))

        with pytest.raises(ValueError, match="one row per masked node"):
            set_node_attr("load", mask, torch.zeros((2, 3)))

    # nodes 0 and 1 belong to the same orbit, which is not allowed for any attribute, even with
    # the same value or with values that are symmetry copies
    two_nodes = torch.zeros(data.num_nodes, dtype=torch.bool)
    two_nodes[[0, 1]] = True

    with pytest.raises(ValueError, match="same orbit"):
        data.set_node_attr_symmetrical("load", two_nodes, torch.tensor([[0., 0., -5.], [0., 0., -5.]]))

    with pytest.raises(ValueError, match="same orbit"):
        data.set_node_attr_symmetrical("coords", two_nodes, torch.tensor([[2., 0., 3.], [0., 2., 3.]]))

    with pytest.raises(ValueError, match="not classified as a transform or copy attribute"):
        data.set_node_attr_symmetrical("stiffness", mask, torch.tensor([[4.]]))

    # rot4 has a single level
    with pytest.raises(ValueError, match="one entry per symmetry level"):
        data.set_node_attr_symmetrical("load", mask, torch.tensor([[0., 0., -5.]]), replicate_levels=[True, False])

    # set_node_attr needs neither one node per orbit nor a transform or copy attribute
    data.set_node_attr("load", two_nodes, torch.tensor([[0., 0., -5.], [0., 0., -5.]]))
    data.set_node_attr("stiffness", mask, torch.tensor([[4.]]))

    assert torch.equal(data.load[:2], torch.tensor([[0., 0., -5.], [0., 0., -5.]]))
    assert torch.equal(data.stiffness[0], torch.tensor([4.]))


def build_nested_structure():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    # level 0: 3 rings stacked along z, level 1: 4 sectors around the z axis
    stacked = data.create_translation(3, torch.tensor([0., 0., 1.]))
    rotation = data.create_rotational_symmetry(4)
    data.add_symmetry({"cylinder": data.combine_symmetry(stacked, rotation)}, copy_attrs=["load"])

    # one orbit of 12 nodes: orbit position p is ring p % 3 of sector p // 3
    data.add_nodes_symmetrical(symmetry="cylinder", coords=torch.tensor([[1., 0., 0.]]), load=torch.zeros((1, 1)))

    return data


def test_set_node_attr_symmetrical_with_replicate_levels():

    # node 4 is ring 1 of sector 1
    mask = torch.zeros(12, dtype=torch.bool)
    mask[4] = True

    # the ring level is frozen: only ring 1 of every sector is updated
    data = build_nested_structure()
    data.set_node_attr_symmetrical("load", mask, torch.tensor([[7.]]), replicate_levels=[False, True])

    assert torch.where(data.load.view(-1) == 7.)[0].tolist() == [1, 4, 7, 10]

    # the sector level is frozen: only the rings of sector 1 are updated
    data = build_nested_structure()
    data.set_node_attr_symmetrical("load", mask, torch.tensor([[7.]]), replicate_levels=[True, False])

    assert torch.where(data.load.view(-1) == 7.)[0].tolist() == [3, 4, 5]

    # every level is frozen: only the selected node is updated
    data = build_nested_structure()
    data.set_node_attr_symmetrical("load", mask, torch.tensor([[7.]]), replicate_levels=[False, False])

    assert torch.where(data.load.view(-1) == 7.)[0].tolist() == [4]

    # a transform attribute is rebuilt from the selected node on ring 1 only, the other rings keep
    # their coords
    data = build_nested_structure()
    coords_before = data.coords.clone()
    data.set_node_attr_symmetrical("coords", mask, torch.tensor([[0., 2., 1.]]), replicate_levels=[False, True])

    ring_1_expected = torch.tensor([[2., 0., 1.], [0., 2., 1.], [-2., 0., 1.], [0., -2., 1.]])
    other_rings = torch.tensor([0, 2, 3, 5, 6, 8, 9, 11])

    assert torch.allclose(data.coords[[1, 4, 7, 10]], ring_1_expected, atol=1e-6)
    assert torch.equal(data.coords[other_rings], coords_before[other_rings])


def test_set_node_attr_symmetrical_replicate_levels_none_matches_all_true():

    mask = torch.zeros(12, dtype=torch.bool)
    mask[4] = True

    default = build_nested_structure()
    default.set_node_attr_symmetrical("coords", mask, torch.tensor([[0., 2., 1.]]))

    all_true = build_nested_structure()
    all_true.set_node_attr_symmetrical("coords", mask, torch.tensor([[0., 2., 1.]]), replicate_levels=[True, True])

    assert torch.equal(default.coords, all_true.coords)


def test_set_node_attr_symmetrical_rejects_several_symmetries():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    data.add_symmetry(
        {"rot4": data.create_rotational_symmetry(4), "mirror": data.create_mirror_symmetry()}, copy_attrs=["load"]
    )

    # nodes 0-3: a rot4 orbit, nodes 4-5: a mirror orbit
    data.add_nodes_symmetrical(symmetry="rot4", coords=torch.tensor([[1., 0., 0.]]), load=torch.zeros((1, 1)))
    data.add_nodes_symmetrical(symmetry="mirror", coords=torch.tensor([[2., 1., 0.]]), load=torch.zeros((1, 1)))

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[[0, 4]] = True

    with pytest.raises(ValueError, match="several registered symmetries"):
        data.set_node_attr_symmetrical("load", mask, torch.tensor([[1.], [2.]]))

    assert torch.equal(data.load, torch.zeros((6, 1)))

    # one call per symmetry
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[0] = True
    data.set_node_attr_symmetrical("load", mask, torch.tensor([[1.]]))

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[4] = True
    data.set_node_attr_symmetrical("load", mask, torch.tensor([[2.]]))

    assert torch.equal(data.load, torch.tensor([[1.]] * 4 + [[2.]] * 2))


def test_set_node_attr_symmetrical_without_symmetry():

    node_attrs = {
        "load": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    data.add_nodes(load=torch.zeros((2, 3)))

    mask = torch.tensor([True, False])

    with pytest.raises(ValueError, match="No symmetry is registered"):
        data.set_node_attr_symmetrical("load", mask, torch.tensor([[1., 2., 3.]]))

    # set_node_attr does not need a symmetry
    data.set_node_attr("load", mask, torch.tensor([[1., 2., 3.]]))

    assert torch.equal(data.load, torch.tensor([[1., 2., 3.], [0., 0., 0.]]))
