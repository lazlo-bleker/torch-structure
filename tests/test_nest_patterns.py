import pytest
import torch
from torch_structure.data.data import StructData


def test_combine_symmetry_matrix_order():

    data = StructData()

    mirror = data.create_mirror_symmetry(normal=torch.tensor([0., 1., 0.]))
    rotation = data.create_rotational_symmetry(4)

    # the mirror is nested inside the rotation: matrix i applies mirror i % 2 first, then rotation i // 2
    symmetry = data.combine_symmetry(mirror, rotation)

    matrices_expected = torch.stack([rotation["matrices"][i // 2] @ mirror["matrices"][i % 2] for i in range(8)])

    seed = torch.tensor([2., 1., 0., 1.])
    points_expected = torch.tensor([
        [2., 1., 0., 1.], [2., -1., 0., 1.], [-1., 2., 0., 1.], [1., 2., 0., 1.],
        [-2., -1., 0., 1.], [-2., 1., 0., 1.], [1., -2., 0., 1.], [-1., -2., 0., 1.],
    ])

    assert symmetry["matrices"].shape == (8, 4, 4)
    assert torch.allclose(symmetry["matrices"], matrices_expected, atol=1e-6)
    assert torch.allclose(symmetry["matrices"] @ seed, points_expected, atol=1e-6)


def test_combine_symmetry_levels():

    data = StructData()

    mirror = data.create_mirror_symmetry(normal=torch.tensor([0., 1., 0.]))
    rotation = data.create_rotational_symmetry(4)

    # mirror inside the rotation: only the identity and the pure mirror are on level 0
    symmetry = data.combine_symmetry(mirror, rotation)

    symmetry_level_expected = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])

    assert torch.equal(symmetry["symmetry_level"], symmetry_level_expected)

    # rotation inside the mirror: the four rotations come first, then their mirrored versions
    symmetry = data.combine_symmetry(rotation, mirror)

    symmetry_level_expected = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    assert torch.equal(symmetry["symmetry_level"], symmetry_level_expected)
    assert torch.allclose(symmetry["matrices"][1], rotation["matrices"][1], atol=1e-6)
    assert torch.allclose(symmetry["matrices"][4], mirror["matrices"][1], atol=1e-6)


def test_combine_symmetry_three_levels():

    data = StructData()

    mirror_x = data.create_mirror_symmetry(normal=torch.tensor([1., 0., 0.]))
    rotation = data.create_rotational_symmetry(3)
    mirror_z = data.create_mirror_symmetry(normal=torch.tensor([0., 0., 1.]))

    symmetry = data.combine_symmetry(data.combine_symmetry(mirror_x, rotation), mirror_z)

    symmetry_level_expected = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])

    assert symmetry["matrices"].shape == (12, 4, 4)
    assert torch.equal(symmetry["symmetry_level"], symmetry_level_expected)
    # matrix 7 applies mirror_x, then the identity rotation, then mirror_z
    assert torch.allclose(symmetry["matrices"][7], mirror_z["matrices"][1] @ mirror_x["matrices"][1], atol=1e-6)
