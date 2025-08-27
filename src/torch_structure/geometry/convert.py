import torch


def point_normal_to_plane(point_normal: torch.Tensor) -> torch.Tensor:
    r"""
    Convert point-normal representation to plane representation.

    Args:
        point_normal (torch.Tensor): Tensor of shape [N, 6] where N is the number of
            points. The first three columns are the point coordinates (x, y, z) and the
            last three columns are the normal vector components (nx, ny, nz).

    Returns:
        (torch.Tensor): Tensor of shape [N, 4] representing the plane coefficients
            (a, b, c, d) in the form ax + by + cz + d = 0.
    """
    point = point_normal[:, :3]
    normal = point_normal[:, 3:]
    a, b, c = normal.unbind(dim=1)
    d = -(a * point[:, 0] + b * point[:, 1] + c * point[:, 2])
    plane = torch.stack((a, b, c, d), dim=1)
    return plane
