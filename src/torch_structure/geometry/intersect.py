import torch
from torch_structure.geometry import point_normal_to_plane


def line_plane_intersect(plane, point, vector, eps=1e-12, return_t=False):
    """
    Vectorized intersection of multiple lines-plane pairs.

    Args:
        plane: Tensor of shape [N, 4] representing the plane coefficients (a, b, c, d)
            in the form ax + by + cz + d = 0. Alternatively, a tensor of shape [N, 6]
            representing the plane in point-normal where the first three columns are the point coordinates (x, y, z)
            and the last three columns are the normal vector components (nx, ny, nz).
            eps:   float, small value to avoid division by zero
        point: Tensor of shape [N, 3] representing the points (x, y, z) of the lines.
        vector: Tensor of shape [N, 3] representing the direction vectors (dx, dy, dz) of the lines.

    Returns:
        intersection point
    """
    if plane.size(-1) == 6:
        plane = point_normal_to_plane(plane)
    if plane.dim() != 2 or plane.size(-1) != 4:
        raise ValueError("Invalid plane representation. Expected [N,4] or [N,6].")
    if point.ndim != 2 or point.size(-1) != 3:
        raise ValueError("r0 must have shape [N,3].")
    if vector.ndim != 2 or vector.size(-1) != 3:
        raise ValueError("v must have shape [N,3].")
    if plane.size(0) != point.size(0) or point.size(0) != vector.size(0):
        raise ValueError("Batch dimension N must match for plane, point, and vector.")

    a, b, c, d = plane.unbind(dim=-1)
    n = torch.stack([a, b, c], dim=-1)

    # Line param: r(t) = r0 + t * v
    r0 = point
    v = vector

    denom = (n * v).sum(dim=-1)
    numer = (n * r0).sum(dim=-1) + d

    is_parallel = denom.abs() <= eps
    is_in_plane = is_parallel & (numer.abs() <= eps)

    t = torch.zeros_like(denom)
    t[~is_parallel] = -numer[~is_parallel] / denom[~is_parallel]

    intersection = r0 + t.unsqueeze(-1) * v

    # Set in plane to r0
    intersection[is_in_plane] = r0[is_in_plane]

    # Set parallel but not in plane to NaN
    no_intersect = is_parallel & (~is_in_plane)
    intersection[no_intersect] = torch.nan

    if return_t:
        return intersection, t
    else:
        return intersection
