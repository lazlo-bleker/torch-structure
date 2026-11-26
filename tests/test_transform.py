import pytest
import torch
from torch_structure.geometry import mirror, rotate, scale, skew, translate


PRECISIONS = [torch.float32, torch.float64]
SKEW_FACTORS = ["x_per_y", "x_per_z", "y_per_x", "y_per_z", "z_per_x", "z_per_y"]

POINTS = [[1., 2., 3.], [-0.5, 0.25, 4.], [2., -1., -3.5]]


@pytest.mark.parametrize("dtype", PRECISIONS)
@pytest.mark.parametrize("param_dtype", PRECISIONS)
def test_rotate_keeps_dtype(param_dtype, dtype):

    coords = torch.tensor(POINTS, dtype=dtype)
    rotation_angle = torch.tensor(0.7, dtype=param_dtype)
    rotation_axis = torch.tensor([1., -2., 0.5], dtype=param_dtype)
    rotation_center = torch.tensor([0.5, -1., 2.], dtype=param_dtype)

    assert rotate(coords, rotation_angle, rotation_axis, rotation_center).dtype == dtype


@pytest.mark.parametrize("dtype", PRECISIONS)
def test_rotate_four_quarter_turns_about_z_is_identity(dtype):

    coords = torch.tensor(POINTS, dtype=dtype)

    rotated = coords
    for _ in range(3):
        rotated = rotate(rotated, torch.pi / 2)
        assert not torch.allclose(rotated, coords)
    rotated = rotate(rotated, torch.pi / 2)

    torch.testing.assert_close(rotated, coords)


@pytest.mark.parametrize("dtype", PRECISIONS)
def test_rotate_full_turn_about_tilted_axis_is_identity(dtype):

    coords = torch.tensor(POINTS, dtype=dtype)
    rotation_axis = torch.tensor([1., -2., 0.5])
    rotation_center = torch.tensor([0.5, -1., 2.])

    rotated = coords
    for _ in range(4):
        rotated = rotate(rotated, 2 * torch.pi / 5, rotation_axis, rotation_center)
        assert not torch.allclose(rotated, coords)
    rotated = rotate(rotated, 2 * torch.pi / 5, rotation_axis, rotation_center)

    torch.testing.assert_close(rotated, coords)


@pytest.mark.parametrize("dtype", PRECISIONS)
@pytest.mark.parametrize("param_dtype", PRECISIONS)
def test_translate_keeps_dtype(param_dtype, dtype):

    coords = torch.tensor(POINTS, dtype=dtype)
    translation_vector = torch.tensor([1., -2., 3.5], dtype=param_dtype)

    assert translate(coords, translation_vector).dtype == dtype


@pytest.mark.parametrize("dtype", PRECISIONS)
def test_translate_and_back_is_identity(dtype):

    coords = torch.tensor(POINTS, dtype=dtype)
    translation_vector = torch.tensor([1., -2., 3.5])

    translated = translate(coords, translation_vector)
    assert not torch.allclose(translated, coords)

    torch.testing.assert_close(translate(translated, -translation_vector), coords)


@pytest.mark.parametrize("dtype", PRECISIONS)
@pytest.mark.parametrize("param_dtype", PRECISIONS)
def test_scale_keeps_dtype(param_dtype, dtype):

    coords = torch.tensor(POINTS, dtype=dtype)

    assert scale(coords, torch.tensor(3., dtype=param_dtype)).dtype == dtype
    assert scale(coords, torch.tensor([2., -0.5, 4.], dtype=param_dtype)).dtype == dtype


@pytest.mark.parametrize("dtype", PRECISIONS)
@pytest.mark.parametrize("scale_factor", [3., torch.tensor([2., -0.5, 4.])], ids=["uniform", "per_axis"])
def test_scale_and_back_is_identity(scale_factor, dtype):

    coords = torch.tensor(POINTS, dtype=dtype)

    scaled = scale(coords, scale_factor)
    assert not torch.allclose(scaled, coords)

    torch.testing.assert_close(scale(scaled, 1 / scale_factor), coords)


@pytest.mark.parametrize("dtype", PRECISIONS)
@pytest.mark.parametrize("param_dtype", PRECISIONS)
def test_skew_keeps_dtype(param_dtype, dtype):

    coords = torch.tensor(POINTS, dtype=dtype)
    factors = {factor: torch.tensor(0.25, dtype=param_dtype) for factor in SKEW_FACTORS}

    assert skew(coords, **factors).dtype == dtype


@pytest.mark.parametrize("dtype", PRECISIONS)
@pytest.mark.parametrize("factor", SKEW_FACTORS)
def test_skew_and_back_is_identity(factor, dtype):

    coords = torch.tensor(POINTS, dtype=dtype)

    skewed = skew(coords, **{factor: 0.75})
    assert not torch.allclose(skewed, coords)

    torch.testing.assert_close(skew(skewed, **{factor: -0.75}), coords)


@pytest.mark.parametrize("dtype", PRECISIONS)
@pytest.mark.parametrize("param_dtype", PRECISIONS)
def test_mirror_keeps_dtype(param_dtype, dtype):

    coords = torch.tensor(POINTS, dtype=dtype)
    plane_normal = torch.tensor([1., 2., -2.], dtype=param_dtype)
    plane_point = torch.tensor([1., 0., 3.], dtype=param_dtype)

    assert mirror(coords, plane_normal, plane_point).dtype == dtype


@pytest.mark.parametrize("dtype", PRECISIONS)
def test_mirror_twice_is_identity(dtype):

    coords = torch.tensor(POINTS, dtype=dtype)

    plane_normal = torch.tensor([0., 0., 1.])

    mirrored = mirror(coords, plane_normal)
    assert not torch.allclose(mirrored, coords)

    torch.testing.assert_close(mirror(mirrored, plane_normal), coords)


@pytest.mark.parametrize("dtype", PRECISIONS)
def test_mirror_twice_across_tilted_plane_is_identity(dtype):

    coords = torch.tensor(POINTS, dtype=dtype)
    plane_normal = torch.tensor([1., 2., -2.])
    plane_point = torch.tensor([1., 0., 3.])

    mirrored = mirror(coords, plane_normal, plane_point)
    assert not torch.allclose(mirrored, coords)

    torch.testing.assert_close(mirror(mirrored, plane_normal, plane_point), coords)
