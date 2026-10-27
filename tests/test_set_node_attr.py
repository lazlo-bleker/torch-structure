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

    # expansion 0 (nodes 0-3) and expansion 1 (nodes 4-7)
    data.add_nodes_by_pattern(
        pattern="rot4",
        coords=torch.tensor([[1., 0., 0.], [2., 0., 0.]]),
        load=torch.tensor([[0., 0., -1.], [0., 0., -2.]]),
    )

    return data


def test_set_node_attr_by_pattern_transform_attribute():

    data = build_structure()

    # select node 2 (expansion 0, position 2): the seed is solved from its new value and the expansion is rebuilt
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[2] = True

    data.set_node_attr_by_pattern("coords", mask, torch.tensor([[-2., 0., 3.]]))

    coords_expected = torch.tensor([
        [2., 0., 3.], [0., 2., 3.], [-2., 0., 3.], [0., -2., 3.],
        [2., 0., 0.], [0., 2., 0.], [-2., 0., 0.], [0., -2., 0.],
    ])

    assert torch.allclose(data.coords, coords_expected, atol=1e-6)


def test_set_node_attr_copied_attribute_and_single_nodes():

    data = build_structure()

    # node 8 is outside any pattern
    data.add_nodes(coords=torch.tensor([[0., 0., 3.]]))

    # load, which is not a transform attribute, set via node 5 (expansion 1, position 1) is copied to the whole expansion
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[5] = True
    data.set_node_attr_by_pattern("load", mask, torch.tensor([[1., 2., 3.]]))

    # set_node_attr only changes node 3
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[3] = True
    data.set_node_attr("load", mask, torch.tensor([[0., 0., -5.]]))

    # a node outside any pattern is set via set_node_attr
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[8] = True
    data.set_node_attr("load", mask, torch.tensor([[0., 0., -7.]]))

    load_expected = torch.tensor([
        [0., 0., -1.], [0., 0., -1.], [0., 0., -1.], [0., 0., -5.],
        [1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.],
        [0., 0., -7.],
    ])

    assert torch.equal(data.load, load_expected)

    # set_node_attr_by_pattern rejects a node outside any pattern, even together with one it can
    # propagate from, and leaves the attribute unchanged
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[[5, 8]] = True

    with pytest.raises(ValueError, match="outside any pattern"):
        data.set_node_attr_by_pattern("load", mask, torch.tensor([[9., 9., 9.], [9., 9., 9.]]))

    assert torch.equal(data.load, load_expected)


def test_set_node_attr_invalid_input():

    data = build_structure()

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[0] = True

    # the checks both versions share
    for set_node_attr in (data.set_node_attr, data.set_node_attr_by_pattern):

        with pytest.raises(ValueError, match="pattern bookkeeping"):
            set_node_attr("expansion_id", mask, torch.tensor([[1]]))

        with pytest.raises(ValueError, match="mask must have 8 entries"):
            set_node_attr("load", torch.zeros(5, dtype=torch.bool), torch.zeros((0, 3)))

        with pytest.raises(ValueError, match="one row per masked node"):
            set_node_attr("load", mask, torch.zeros((2, 3)))

    # nodes 0 and 1 belong to the same expansion, which is not allowed for any attribute, even with
    # the same value or with values that are pattern copies
    two_nodes = torch.zeros(data.num_nodes, dtype=torch.bool)
    two_nodes[[0, 1]] = True

    with pytest.raises(ValueError, match="same expansion"):
        data.set_node_attr_by_pattern("load", two_nodes, torch.tensor([[0., 0., -5.], [0., 0., -5.]]))

    with pytest.raises(ValueError, match="same expansion"):
        data.set_node_attr_by_pattern("coords", two_nodes, torch.tensor([[2., 0., 3.], [0., 2., 3.]]))

    # rot4 has a single level
    with pytest.raises(ValueError, match="one entry per pattern level"):
        data.set_node_attr_by_pattern("load", mask, torch.tensor([[0., 0., -5.]]), replicate_levels=[True, False])

    # set_node_attr does not need one node per expansion
    data.set_node_attr("load", two_nodes, torch.tensor([[0., 0., -5.], [0., 0., -5.]]))

    assert torch.equal(data.load[:2], torch.tensor([[0., 0., -5.], [0., 0., -5.]]))


def build_nested_structure():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    # level 0: 3 rings stacked along z, level 1: 4 sectors around the z axis
    stacked = data.create_translational_pattern(3, torch.tensor([0., 0., 1.]))
    rotation = data.create_rotational_pattern(4)
    data.add_patterns({"cylinder": data.nest_patterns([stacked, rotation])})

    # one expansion of 12 nodes: expansion position p is ring p % 3 of sector p // 3
    data.add_nodes_by_pattern(pattern="cylinder", coords=torch.tensor([[1., 0., 0.]]), load=torch.zeros((1, 1)))

    return data


def test_set_node_attr_by_pattern_with_replicate_levels():

    # node 4 is ring 1 of sector 1
    mask = torch.zeros(12, dtype=torch.bool)
    mask[4] = True

    # the ring level is frozen: only ring 1 of every sector is updated
    data = build_nested_structure()
    data.set_node_attr_by_pattern("load", mask, torch.tensor([[7.]]), replicate_levels=[False, True])

    assert torch.where(data.load.view(-1) == 7.)[0].tolist() == [1, 4, 7, 10]

    # the sector level is frozen: only the rings of sector 1 are updated
    data = build_nested_structure()
    data.set_node_attr_by_pattern("load", mask, torch.tensor([[7.]]), replicate_levels=[True, False])

    assert torch.where(data.load.view(-1) == 7.)[0].tolist() == [3, 4, 5]

    # every level is frozen: only the selected node is updated
    data = build_nested_structure()
    data.set_node_attr_by_pattern("load", mask, torch.tensor([[7.]]), replicate_levels=[False, False])

    assert torch.where(data.load.view(-1) == 7.)[0].tolist() == [4]

    # a transform attribute is rebuilt from the selected node on ring 1 only, the other rings keep
    # their coords
    data = build_nested_structure()
    coords_before = data.coords.clone()
    data.set_node_attr_by_pattern("coords", mask, torch.tensor([[0., 2., 1.]]), replicate_levels=[False, True])

    ring_1_expected = torch.tensor([[2., 0., 1.], [0., 2., 1.], [-2., 0., 1.], [0., -2., 1.]])
    other_rings = torch.tensor([0, 2, 3, 5, 6, 8, 9, 11])

    assert torch.allclose(data.coords[[1, 4, 7, 10]], ring_1_expected, atol=1e-6)
    assert torch.equal(data.coords[other_rings], coords_before[other_rings])


def test_set_node_attr_by_pattern_replicate_levels_none_matches_all_true():

    mask = torch.zeros(12, dtype=torch.bool)
    mask[4] = True

    default = build_nested_structure()
    default.set_node_attr_by_pattern("coords", mask, torch.tensor([[0., 2., 1.]]))

    all_true = build_nested_structure()
    all_true.set_node_attr_by_pattern("coords", mask, torch.tensor([[0., 2., 1.]]), replicate_levels=[True, True])

    assert torch.equal(default.coords, all_true.coords)


def test_set_node_attr_by_pattern_rejects_several_patterns():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 1), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    data.add_patterns({"rot4": data.create_rotational_pattern(4), "mirror": data.create_mirror_pattern()})

    # nodes 0-3: a rot4 expansion, nodes 4-5: a mirror expansion
    data.add_nodes_by_pattern(pattern="rot4", coords=torch.tensor([[1., 0., 0.]]), load=torch.zeros((1, 1)))
    data.add_nodes_by_pattern(pattern="mirror", coords=torch.tensor([[2., 1., 0.]]), load=torch.zeros((1, 1)))

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[[0, 4]] = True

    with pytest.raises(ValueError, match="several registered patterns"):
        data.set_node_attr_by_pattern("load", mask, torch.tensor([[1.], [2.]]))

    assert torch.equal(data.load, torch.zeros((6, 1)))

    # one call per pattern
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[0] = True
    data.set_node_attr_by_pattern("load", mask, torch.tensor([[1.]]))

    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[4] = True
    data.set_node_attr_by_pattern("load", mask, torch.tensor([[2.]]))

    assert torch.equal(data.load, torch.tensor([[1.]] * 4 + [[2.]] * 2))


def test_set_node_attr_by_pattern_without_pattern():

    node_attrs = {
        "load": torch.empty((0, 3), dtype=torch.float),
    }

    data = StructData(node_attrs=node_attrs)

    data.add_nodes(load=torch.zeros((2, 3)))

    mask = torch.tensor([True, False])

    with pytest.raises(ValueError, match="No pattern is registered"):
        data.set_node_attr_by_pattern("load", mask, torch.tensor([[1., 2., 3.]]))

    # set_node_attr does not need a pattern
    data.set_node_attr("load", mask, torch.tensor([[1., 2., 3.]]))

    assert torch.equal(data.load, torch.tensor([[1., 2., 3.], [0., 0., 0.]]))
