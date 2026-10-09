import pytest
import torch
from torch_structure.data.data import StructData


def test_create_rotational_symmetry_about_z_axis():

    data = StructData()

    symmetry = data.create_rotational_symmetry(4)

    # quarter turns, counter-clockwise when looking down the z-axis: (1, 0, 0) -> (0, 1, 0)
    matrices_expected = torch.tensor([
        [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
        [[0., -1., 0., 0.], [1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
        [[-1., 0., 0., 0.], [0., -1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
        [[0., 1., 0., 0.], [-1., 0., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
    ])
    symmetry_level_expected = torch.zeros(4, dtype=torch.long)

    assert symmetry["matrices"].shape == (4, 4, 4)
    assert torch.allclose(symmetry["matrices"], matrices_expected, atol=1e-6)
    assert torch.equal(symmetry["matrices"][0], torch.eye(4))
    assert torch.equal(symmetry["symmetry_level"], symmetry_level_expected)


def test_create_rotational_symmetry_with_origin_and_axis():

    data = StructData()

    # half turn about the vertical line through (1, 0, 0); the axis does not need to be normalized
    symmetry = data.create_rotational_symmetry(
        2, origin=torch.tensor([1., 0., 0.]), rotation_axis=torch.tensor([0., 0., 2.])
    )

    matrix_expected = torch.tensor([[-1., 0., 0., 2.], [0., -1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    origin = torch.tensor([1., 0., 0., 1.])

    assert symmetry["matrices"].shape == (2, 4, 4)
    assert torch.allclose(symmetry["matrices"][1], matrix_expected, atol=1e-6)
    assert torch.allclose(symmetry["matrices"][1] @ origin, origin, atol=1e-6)

    # quarter turns about the x-axis: (0, 1, 0) -> (0, 0, 1)
    symmetry = data.create_rotational_symmetry(4, rotation_axis=torch.tensor([1., 0., 0.]))

    point = torch.tensor([0., 1., 0., 1.])
    point_expected = torch.tensor([0., 0., 1., 1.])

    assert torch.allclose(symmetry["matrices"][1] @ point, point_expected, atol=1e-6)


def test_create_rotational_symmetry_invalid_n():

    data = StructData()

    with pytest.raises(ValueError, match="n must be at least 1"):
        data.create_rotational_symmetry(0)

    with pytest.raises(ValueError, match="n must be at least 1"):
        data.create_rotational_symmetry(-2)
