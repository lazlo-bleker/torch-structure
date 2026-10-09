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


def test_add_symmetry_registers_symmetry():

    data = build_structure()

    rotation = data.create_rotational_symmetry(4)
    data.add_symmetry({"rot4": rotation}, copy_attrs=["load", "force"])

    name_to_symmetry_expected = {"rot4": 0}
    symmetry_matrix_id_expected = torch.zeros(4, dtype=torch.long)
    symmetry_level_expected = torch.zeros(4, dtype=torch.long)

    assert data.metadata["name_to_symmetry"] == name_to_symmetry_expected
    assert torch.equal(data.symmetry_matrices, rotation["matrices"])
    assert torch.equal(data.symmetry_matrix_id, symmetry_matrix_id_expected)
    assert torch.equal(data.symmetry_level, symmetry_level_expected)

    # transform attributes default to coords; copy attributes can be node and edge attributes
    assert data.metadata["symmetry_transform_attrs"][0] == ["coords"]
    assert data.metadata["symmetry_copy_attrs"][0] == ["load", "force"]

    # the first symmetry adds the bookkeeping attributes
    for attr in ("orbit_id", "orbit_position", "symmetry_id"):
        assert attr in data.metadata["node_attr_list"]
        assert getattr(data, attr).shape == (0, 1)

    for attr in ("symmetry_matrices", "symmetry_matrix_id", "symmetry_level"):
        assert attr in data.metadata["graph_attr_list"]


def test_add_symmetry_several_symmetries_with_existing_nodes():

    data = build_structure()

    data.add_nodes(coords=torch.tensor([[0., 0., 0.], [1., 1., 1.]]))

    rotation = data.create_rotational_symmetry(4)
    mirror = data.create_mirror_symmetry()
    data.add_symmetry({"rot4": rotation, "mirror": mirror}, copy_attrs=["load"])
    data.add_symmetry({"d4": data.combine_symmetry(mirror, rotation)}, transform_attrs=[])

    name_to_symmetry_expected = {"rot4": 0, "mirror": 1, "d4": 2}
    symmetry_matrix_id_expected = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
    symmetry_level_expected = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    transform_attrs_expected = {0: ["coords"], 1: ["coords"], 2: []}
    copy_attrs_expected = {0: ["load"], 1: ["load"], 2: []}
    bookkeeping_expected = torch.tensor([[-1], [-1]])

    assert data.metadata["name_to_symmetry"] == name_to_symmetry_expected
    assert data.symmetry_matrices.shape == (14, 4, 4)
    assert torch.equal(data.symmetry_matrix_id, symmetry_matrix_id_expected)
    assert torch.equal(data.symmetry_level, symmetry_level_expected)
    assert data.metadata["symmetry_transform_attrs"] == transform_attrs_expected
    assert data.metadata["symmetry_copy_attrs"] == copy_attrs_expected

    # nodes that existed before the first symmetry do not belong to any symmetry
    assert torch.equal(data.orbit_id, bookkeeping_expected)
    assert torch.equal(data.orbit_position, bookkeeping_expected)
    assert torch.equal(data.symmetry_id, bookkeeping_expected)


def test_add_symmetry_invalid_input():

    data = build_structure()

    rotation = data.create_rotational_symmetry(4)
    data.add_symmetry({"rot4": rotation})

    with pytest.raises(ValueError, match="already exists"):
        data.add_symmetry({"rot4": rotation})

    with pytest.raises(ValueError, match="symmetry bookkeeping"):
        data.add_symmetry({"rot4_b": rotation}, copy_attrs=["orbit_id"])

    with pytest.raises(ValueError, match="registered edge attribute"):
        data.add_symmetry({"rot4_b": rotation}, transform_attrs=["force"])

    with pytest.raises(ValueError, match="one entry per matrix"):
        data.add_symmetry({"rot4_b": {"matrices": rotation["matrices"], "symmetry_level": torch.zeros(3, dtype=torch.long)}})

    # the failed calls did not register anything
    assert data.metadata["name_to_symmetry"] == {"rot4": 0}
    assert data.symmetry_matrices.shape == (4, 4, 4)
