import torch
from torch_structure.geometry.intersect import (
    point_normal_to_plane,
    line_plane_intersect,
)


def test_point_normal_to_plane():
    pn = torch.tensor(
        [
            [0.0, 0.0, 5.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, -5.0, 6.0],
        ]
    )
    plane = point_normal_to_plane(pn)
    correct_plane = torch.tensor(
        [
            [0.0, 0.0, 1.0, -5.0],
            [4.0, -5.0, 6.0, -12.0],
        ]
    )
    assert plane.shape == (2, 4)
    assert torch.allclose(plane, correct_plane, atol=1e-6)

    # Sanity: original points must satisfy ax + by + cz + d = 0
    p = pn[:, :3]
    lhs = (plane[:, :3] * p).sum(dim=-1) + plane[:, 3]
    assert torch.allclose(lhs, torch.zeros_like(lhs), atol=1e-6)


def test_line_plane_intersect():
    # Case A: simple hit
    # Plane: z = 0
    plane = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    point = torch.tensor([[0.0, 0.0, 1.0]])
    vector = torch.tensor([[0.0, 0.0, -1.0]])
    out = line_plane_intersect(plane, point, vector)
    intersection = torch.tensor([[0.0, 0.0, 0.0]])
    assert torch.allclose(out, intersection, atol=1e-6)

    # Case B: hit exactly at r0 (numerator == 0, denom != 0)
    # Plane: x + y + z - 1 = 0
    plane = torch.tensor([[1.0, 1.0, 1.0, -1.0]])
    point = torch.tensor([[1.0, 0.0, 0.0]])  # lies on plane (1+0+0-1=0)
    vector = torch.tensor([[1.0, 1.0, 1.0]])  # not parallel to plane (denom = 3)
    out = line_plane_intersect(plane, point, vector)
    assert torch.allclose(out, point, atol=1e-6)

    # Case C: parallel, not in plane -> NaNs
    # Plane: z = 0 ; Line: z=1, direction parallel to plane
    plane = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    point = torch.tensor([[0.0, 0.0, 1.0]])
    vector = torch.tensor([[1.0, 0.0, 0.0]])  # n·v = 0
    out = line_plane_intersect(plane, point, vector)
    assert torch.isnan(out).all()

    # Case D: line lies in the plane (parallel & in-plane) -> return r0
    # Plane: z = 0 ; Line: z=0, direction parallel to plane
    plane = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    point = torch.tensor([[1.0, 2.0, 0.0]])
    vector = torch.tensor([[1.0, 0.0, 0.0]])  # n·v = 0, n·r0 + d = 0
    out = line_plane_intersect(plane, point, vector)
    assert torch.allclose(out, point, atol=1e-6)

    # Case E: point-normal plane format [N,6]
    # Plane: point (0,0,0), normal (0,0,1) => z=0
    plane_pn = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    point = torch.tensor([[0.0, 0.0, 5.0]])
    vector = torch.tensor([[0.0, 0.0, -2.0]])
    out = line_plane_intersect(plane_pn, point, vector)
    intersection = torch.tensor([[0.0, 0.0, 0.0]])
    assert torch.allclose(out, intersection, atol=1e-6)

    # Case F: vectorized batch with mixed outcomes
    # 0: hit; 1: in-plane; 2: no-intersect (parallel off-plane)
    plane = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],  # z=0
            [0.0, 0.0, 1.0, 0.0],  # z=0
            [0.0, 0.0, 1.0, 0.0],  # z=0
        ]
    )
    point = torch.tensor(
        [
            [0.0, 0.0, 2.0],  # above plane
            [1.0, 2.0, 0.0],  # on plane
            [0.0, 0.0, 1.0],  # off plane
        ]
    )
    vector = torch.tensor(
        [
            [0.0, 0.0, -1.0],  # intersects
            [1.0, 0.0, 0.0],  # in-plane direction
            [1.0, 0.0, 0.0],  # parallel to plane, off-plane -> NaN
        ]
    )
    out = line_plane_intersect(plane, point, vector)
    assert torch.allclose(out[0], torch.tensor([0.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(out[1], point[1], atol=1e-6)
    assert torch.isnan(out[2]).all()
