import pytest
import torch
from torch_structure.data.data import StructData


def test_create_mirror_pattern_default_plane():

    data = StructData()

    pattern = data.create_mirror_pattern()

    # the identity and the reflection across the plane x = 0
    matrices_expected = torch.tensor([
        [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
        [[-1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]],
    ])
    pattern_level_expected = torch.zeros(2, dtype=torch.long)

    assert torch.equal(pattern["matrices"], matrices_expected)
    assert torch.equal(pattern["pattern_level"], pattern_level_expected)


def test_create_mirror_pattern_with_origin_and_normal():

    data = StructData()

    # plane z = 2; the normal does not need to be normalized
    pattern = data.create_mirror_pattern(origin=torch.tensor([0., 0., 2.]), normal=torch.tensor([0., 0., 3.]))

    matrix_expected = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., -1., 4.], [0., 0., 0., 1.]])
    point = torch.tensor([1., 1., 5., 1.])
    point_expected = torch.tensor([1., 1., -1., 1.])
    point_on_plane = torch.tensor([3., 4., 2., 1.])

    assert torch.equal(pattern["matrices"][1], matrix_expected)
    assert torch.equal(pattern["matrices"][1] @ point, point_expected)
    assert torch.equal(pattern["matrices"][1] @ point_on_plane, point_on_plane)


def test_create_mirror_pattern_diagonal_plane():

    data = StructData()

    # plane x + y = 0 through the origin
    pattern = data.create_mirror_pattern(normal=torch.tensor([1., 1., 0.]))
    mirror = pattern["matrices"][1]

    point = torch.tensor([1., 0., 0., 1.])
    point_expected = torch.tensor([0., -1., 0., 1.])

    assert torch.allclose(mirror @ point, point_expected, atol=1e-6)
    # reflecting twice gives the identity
    assert torch.allclose(mirror @ mirror, torch.eye(4), atol=1e-6)
