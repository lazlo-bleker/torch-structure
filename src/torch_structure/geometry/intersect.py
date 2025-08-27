import torch
from torch_structure.geometry import point_normal_to_plane


def line_plane_intersect(
    plane: torch.Tensor,
    point: torch.Tensor,
    vector: torch.Tensor,
    eps: float = 1e-12,
    return_t: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized intersection of multiple line-plane pairs.

    Args:
        plane (torch.Tensor): shape [N, 4] plane coefficients (a, b, c, d) in the form
            ax + by + cz + d = 0. Alternatively, shape [N, 6] representing the plane in
            point-normal where the first three columns are the point coordinates
            (x, y, z) and the last three columns are the normal vector components
            (nx, ny, nz).
        point (torch.Tensor): shape [N, 3] representing the points (x, y, z) of the
            lines.
        vector (torch.Tensor): shape [N, 3] representing the direction vectors
            (dx, dy, dz) of the lines.
        eps (float, optional): tolerance to avoid division by zero. Defaults to 1e-12.
        return_t (bool, optional): If True, also return line parameters ``t``. Defaults
            to False.

    Returns:
        (torch.Tensor | tuple[torch.Tensor, torch.Tensor]): If ``return_t`` is False:
            intersection points of shape [N, 3]. If ``return_t`` is True: intersection
            points of shape [N, 3], line parameters of shape [N].
    """
    if plane.size(-1) == 6:
        plane = point_normal_to_plane(plane)
    if plane.dim() != 2 or plane.size(-1) != 4:
        raise ValueError("Invalid plane representation. Expected [N, 4] or [N, 6].")
    if point.ndim != 2 or point.size(-1) != 3:
        raise ValueError("point must have shape [N, 3].")
    if vector.ndim != 2 or vector.size(-1) != 3:
        raise ValueError("vector must have shape [N, 3].")
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
