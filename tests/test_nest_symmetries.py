import pytest
import torch
from torch_structure.data.data import StructData


def test_nest_symmetries_matrix_order():

    data = StructData()

    mirror = data.create_mirror_symmetry(normal=torch.tensor([0., 1., 0.]))
    rotation = data.create_rotational_symmetry(4)

    # the mirror is nested inside the rotation: matrix i applies mirror i % 2 first, then rotation i // 2
    symmetry = data.nest_symmetries([mirror, rotation])

    matrices_expected = torch.stack([rotation["matrices"][i // 2] @ mirror["matrices"][i % 2] for i in range(8)])

    seed = torch.tensor([2., 1., 0., 1.])
    points_expected = torch.tensor([
        [2., 1., 0., 1.], [2., -1., 0., 1.], [-1., 2., 0., 1.], [1., 2., 0., 1.],
        [-2., -1., 0., 1.], [-2., 1., 0., 1.], [1., -2., 0., 1.], [-1., -2., 0., 1.],
    ])

    assert symmetry["matrices"].shape == (8, 4, 4)
    assert torch.allclose(symmetry["matrices"], matrices_expected, atol=1e-6)
    assert torch.allclose(symmetry["matrices"] @ seed, points_expected, atol=1e-6)


def test_nest_symmetries_levels():

    data = StructData()

    mirror = data.create_mirror_symmetry(normal=torch.tensor([0., 1., 0.]))
    rotation = data.create_rotational_symmetry(4)

    # mirror inside the rotation: only the identity and the pure mirror are on level 0
    symmetry = data.nest_symmetries([mirror, rotation])

    symmetry_level_expected = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])

    assert torch.equal(symmetry["symmetry_level"], symmetry_level_expected)

    # rotation inside the mirror: the four rotations come first, then their mirrored versions
    symmetry = data.nest_symmetries([rotation, mirror])

    symmetry_level_expected = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    assert torch.equal(symmetry["symmetry_level"], symmetry_level_expected)
    assert torch.allclose(symmetry["matrices"][1], rotation["matrices"][1], atol=1e-6)
    assert torch.allclose(symmetry["matrices"][4], mirror["matrices"][1], atol=1e-6)


def test_nest_symmetries_three_levels():

    data = StructData()

    mirror_x = data.create_mirror_symmetry(normal=torch.tensor([1., 0., 0.]))
    rotation = data.create_rotational_symmetry(3)
    mirror_z = data.create_mirror_symmetry(normal=torch.tensor([0., 0., 1.]))

    symmetry = data.nest_symmetries([mirror_x, rotation, mirror_z])

    symmetry_level_expected = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])

    assert symmetry["matrices"].shape == (12, 4, 4)
    assert torch.equal(symmetry["symmetry_level"], symmetry_level_expected)
    # matrix 7 applies mirror_x, then the identity rotation, then mirror_z
    assert torch.allclose(symmetry["matrices"][7], mirror_z["matrices"][1] @ mirror_x["matrices"][1], atol=1e-6)


def test_nest_symmetries_is_associative():

    data = StructData()

    inner = data.create_rotational_symmetry(4, closes=False)
    middle = data.create_mirror_symmetry(normal=torch.tensor([0., 0., 1.]))
    outer = data.create_translational_symmetry(3, torch.tensor([10., 0., 0.]))

    flat = data.nest_symmetries([inner, middle, outer])
    nested_inner = data.nest_symmetries([data.nest_symmetries([inner, middle]), outer])
    nested_outer = data.nest_symmetries([inner, data.nest_symmetries([middle, outer])])

    for symmetry in (nested_inner, nested_outer):
        assert torch.allclose(symmetry["matrices"], flat["matrices"], atol=1e-6)
        assert torch.equal(symmetry["symmetry_level"], flat["symmetry_level"])
        assert torch.equal(symmetry["closes"], flat["closes"])


def test_nest_symmetries_single_symmetry_is_unchanged():

    data = StructData()

    rotation = data.create_rotational_symmetry(4)

    assert data.nest_symmetries([rotation]) is rotation


def test_nest_symmetries_empty_raises():

    data = StructData()

    with pytest.raises(ValueError, match="at least one symmetry"):
        data.nest_symmetries([])
