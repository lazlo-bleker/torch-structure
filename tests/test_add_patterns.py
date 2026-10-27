import pytest
import torch
from torch_structure.data.data import StructData


def build_structure():

    node_attrs = {
        "coords": torch.empty((0, 3), dtype=torch.float),
        "load": torch.empty((0, 3), dtype=torch.float),
    }

    edge_attrs = {
        "force": torch.empty((0, 1), dtype=torch.float),
    }

    return StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)


def test_add_patterns_registers_pattern():

    data = build_structure()

    rotation = data.create_rotational_pattern(4)
    data.add_patterns({"rot4": rotation})

    name_to_pattern_expected = {"rot4": 0}
    pattern_matrix_id_expected = torch.zeros(4, dtype=torch.long)
    pattern_level_expected = torch.zeros(4, dtype=torch.long)

    assert data.metadata["name_to_pattern"] == name_to_pattern_expected
    assert torch.equal(data.pattern_matrices, rotation["matrices"])
    assert torch.equal(data.pattern_matrix_id, pattern_matrix_id_expected)
    assert torch.equal(data.pattern_level, pattern_level_expected)

    # transform attributes default to coords
    assert data.metadata["pattern_transform_attrs"][0] == ["coords"]

    # the first pattern adds the bookkeeping attributes
    for attr in ("expansion_id", "expansion_position", "pattern_id"):
        assert attr in data.metadata["node_attr_list"]
        assert getattr(data, attr).shape == (0, 1)

    for attr in ("pattern_matrices", "pattern_matrix_id", "pattern_level"):
        assert attr in data.metadata["graph_attr_list"]


def test_add_patterns_several_patterns_with_existing_nodes():

    data = build_structure()

    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [1., 1., 1.]]))

    rotation = data.create_rotational_pattern(4)
    mirror = data.create_mirror_pattern()
    data.add_patterns({"rot4": rotation, "mirror": mirror})
    data.add_patterns({"d4": data.nest_patterns([mirror, rotation])}, transform_attrs=[])

    name_to_pattern_expected = {"rot4": 0, "mirror": 1, "d4": 2}
    pattern_matrix_id_expected = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
    pattern_level_expected = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    transform_attrs_expected = {0: ["coords"], 1: ["coords"], 2: []}
    bookkeeping_expected = torch.tensor([[-1], [-1]])

    assert data.metadata["name_to_pattern"] == name_to_pattern_expected
    assert data.pattern_matrices.shape == (14, 4, 4)
    assert torch.equal(data.pattern_matrix_id, pattern_matrix_id_expected)
    assert torch.equal(data.pattern_level, pattern_level_expected)
    assert data.metadata["pattern_transform_attrs"] == transform_attrs_expected

    # nodes that existed before the first pattern do not belong to any pattern
    assert torch.equal(data.expansion_id, bookkeeping_expected)
    assert torch.equal(data.expansion_position, bookkeeping_expected)
    assert torch.equal(data.pattern_id, bookkeeping_expected)


def test_add_patterns_keeps_its_own_transform_attrs():

    data = build_structure()

    rotation = data.create_rotational_pattern(4)
    mirror = data.create_mirror_pattern()
    transform_attrs = ["coords"]
    data.add_patterns({"rot4": rotation, "mirror": mirror}, transform_attrs=transform_attrs)

    # changing the given list afterwards does not change the registered patterns
    transform_attrs.append("load")

    assert data.metadata["pattern_transform_attrs"] == {0: ["coords"], 1: ["coords"]}


def test_add_patterns_invalid_input():

    data = build_structure()

    rotation = data.create_rotational_pattern(4)
    data.add_patterns({"rot4": rotation})

    with pytest.raises(ValueError, match="already exists"):
        data.add_patterns({"rot4": rotation})

    with pytest.raises(ValueError, match="pattern bookkeeping"):
        data.add_patterns({"rot4_b": rotation}, transform_attrs=["expansion_id"])

    with pytest.raises(ValueError, match="registered edge attribute"):
        data.add_patterns({"rot4_b": rotation}, transform_attrs=["force"])

    with pytest.raises(ValueError, match="one entry per matrix"):
        data.add_patterns({"rot4_b": {"matrices": rotation["matrices"], "pattern_level": torch.zeros(3, dtype=torch.long)}})

    # the failed calls did not register anything
    assert data.metadata["name_to_pattern"] == {"rot4": 0}
    assert data.pattern_matrices.shape == (4, 4, 4)
