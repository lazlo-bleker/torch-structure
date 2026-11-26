import torch


def rotate(coords, rotation_angle, rotation_axis=torch.tensor([0, 0, 1]), rotation_center=torch.tensor([0, 0, 0])):
    r""" Rotate points around an axis through a rotation center.

    Args:
        coords (torch.Tensor): Tensor of shape [N, 3] representing the points to rotate.
        rotation_angle (float | torch.Tensor): Scalar rotation angle in radians, following the right-hand rule
            around ``rotation_axis``.
        rotation_axis (torch.Tensor, optional): Tensor of shape [3] representing the direction of the rotation axis.
            Does not need to be normalized. Defaults to the z-axis.
        rotation_center (torch.Tensor, optional): Tensor of shape [3] representing a point on the rotation axis.
            Defaults to the origin.

    Returns:
        (torch.Tensor): Tensor of shape [N, 3] representing the rotated points.
    """
    if coords.ndim != 2 or coords.size(-1) != 3:
        raise ValueError(f"coords must have shape [N, 3], got {list(coords.shape)}.")
    if not coords.is_floating_point():
        coords = coords.to(torch.get_default_dtype())
    rotation_angle = torch.as_tensor(rotation_angle, dtype=coords.dtype, device=coords.device)
    rotation_axis = torch.as_tensor(rotation_axis, dtype=coords.dtype, device=coords.device)
    rotation_center = torch.as_tensor(rotation_center, dtype=coords.dtype, device=coords.device)

    if rotation_angle.ndim != 0:
        raise ValueError(f"rotation_angle must be a single number, got shape {list(rotation_angle.shape)}.")
    if rotation_axis.shape != (3,):
        raise ValueError(f"rotation_axis must have shape [3], got {list(rotation_axis.shape)}.")
    if rotation_center.shape != (3,):
        raise ValueError(f"rotation_center must have shape [3], got {list(rotation_center.shape)}.")
    axis_length = torch.linalg.vector_norm(rotation_axis)
    if axis_length == 0:
        raise ValueError("Can not rotate around the axis [0,0,0]. At least one dimension has to be non-zero.")

    rotation_axis = rotation_axis / axis_length
    
    r_x, r_y, r_z = rotation_axis.unbind()
    c = torch.cos(rotation_angle)
    s = torch.sin(rotation_angle)

    rotation_matrix = torch.stack([
        torch.stack([r_x**2 * (1 - c) + c, r_x * r_y * (1 - c) - r_z * s, r_x * r_z * (1 - c) + r_y * s]),
        torch.stack([r_x * r_y * (1 - c) + r_z * s, r_y**2 * (1 - c) + c, r_y * r_z * (1 - c) - r_x * s]),
        torch.stack([r_x * r_z * (1 - c) - r_y * s, r_y * r_z * (1 - c) + r_x * s, r_z**2 * (1 - c) + c]),
    ])

    coords = coords - rotation_center

    coords = coords @ rotation_matrix.T

    coords = coords + rotation_center

    return coords



def translate(coords, translation_vector):
    r""" Translate points by a translation vector.

    Args:
        coords (torch.Tensor): Tensor of shape [N, 3] representing the points to translate.
        translation_vector (torch.Tensor): Tensor of shape [3] representing the translation vector.

    Returns:
        (torch.Tensor): Tensor of shape [N, 3] representing the translated points.
    """
    if coords.ndim != 2 or coords.size(-1) != 3:
        raise ValueError(f"coords must have shape [N, 3], got {list(coords.shape)}.")
    if not coords.is_floating_point():
        coords = coords.to(torch.get_default_dtype())
    translation_vector = torch.as_tensor(translation_vector, dtype=coords.dtype, device=coords.device)

    if translation_vector.shape != (3,):
        raise ValueError(f"translation_vector must have shape [3], got {list(translation_vector.shape)}.")

    return coords + translation_vector


def scale(coords, scale_factor):
    r""" Scale points about the origin, either uniformly or separately along x, y and z.

    Args:
        coords (torch.Tensor): Tensor of shape [N, 3] representing the points to scale.
        scale_factor (float | torch.Tensor): A single number for uniform scaling, or a tensor of shape [3] with one
            factor each for x, y and z. A negative factor mirrors the points along the respective axis.

    Returns:
        (torch.Tensor): Tensor of shape [N, 3] representing the scaled points.
    """
    if coords.ndim != 2 or coords.size(-1) != 3:
        raise ValueError(f"coords must have shape [N, 3], got {list(coords.shape)}.")
    if not coords.is_floating_point():
        coords = coords.to(torch.get_default_dtype())
    scale_factor = torch.as_tensor(scale_factor, dtype=coords.dtype, device=coords.device)

    if scale_factor.ndim != 0 and scale_factor.shape != (3,):
        raise ValueError(f"scale_factor must be a single number or have shape [3], got {list(scale_factor.shape)}.")

    return coords * scale_factor


def skew(coords, x_per_y=0, x_per_z=0, y_per_x=0, y_per_z=0, z_per_x=0, z_per_y=0):
    r""" Skew (shear) points about the origin.

    Each factor moves one coordinate in proportion to another; e.g. ``x_per_z`` is how far x moves per unit of z:

    ```
    x' = x + x_per_y * y + x_per_z * z
    y' = y + y_per_x * x + y_per_z * z
    z' = z + z_per_x * x + z_per_y * y
    ```

    Args:
        coords (torch.Tensor): Tensor of shape [N, 3] representing the points to skew.
        x_per_y (float | torch.Tensor, optional): Shift of x per unit of y. Defaults to 0.
        x_per_z (float | torch.Tensor, optional): Shift of x per unit of z. Defaults to 0.
        y_per_x (float | torch.Tensor, optional): Shift of y per unit of x. Defaults to 0.
        y_per_z (float | torch.Tensor, optional): Shift of y per unit of z. Defaults to 0.
        z_per_x (float | torch.Tensor, optional): Shift of z per unit of x. Defaults to 0.
        z_per_y (float | torch.Tensor, optional): Shift of z per unit of y. Defaults to 0.

    Returns:
        (torch.Tensor): Tensor of shape [N, 3] representing the skewed points.
    """
    if coords.ndim != 2 or coords.size(-1) != 3:
        raise ValueError(f"coords must have shape [N, 3], got {list(coords.shape)}.")
    if not coords.is_floating_point():
        coords = coords.to(torch.get_default_dtype())

    factors = []
    for name, factor in [("x_per_y", x_per_y), ("x_per_z", x_per_z), ("y_per_x", y_per_x),
                         ("y_per_z", y_per_z), ("z_per_x", z_per_x), ("z_per_y", z_per_y)]:
        factor = torch.as_tensor(factor, dtype=coords.dtype, device=coords.device)
        if factor.ndim != 0:
            raise ValueError(f"{name} must be a single number, got shape {list(factor.shape)}.")
        factors.append(factor)
    x_per_y, x_per_z, y_per_x, y_per_z, z_per_x, z_per_y = factors

    x, y, z = coords.unbind(dim=1)

    return torch.stack([
        x + x_per_y * y + x_per_z * z,
        y + y_per_x * x + y_per_z * z,
        z + z_per_x * x + z_per_y * y,
    ], dim=1)


def mirror(coords, plane_normal, plane_point=torch.tensor([0, 0, 0])):
    r""" Mirror points across a plane given by its normal and a point on it.

    Args:
        coords (torch.Tensor): Tensor of shape [N, 3] representing the points to mirror.
        plane_normal (torch.Tensor): Tensor of shape [3] representing the normal of the mirror plane. Does not need
            to be normalized. E.g. ``[1, 0, 0]`` gives a plane parallel to the yz-plane, so only x changes.
        plane_point (torch.Tensor, optional): Tensor of shape [3] representing any point on the mirror plane; every
            point on the plane gives the same result. Defaults to the origin.

    Returns:
        (torch.Tensor): Tensor of shape [N, 3] representing the mirrored points.
    """
    if coords.ndim != 2 or coords.size(-1) != 3:
        raise ValueError(f"coords must have shape [N, 3], got {list(coords.shape)}.")
    if not coords.is_floating_point():
        coords = coords.to(torch.get_default_dtype())
    plane_normal = torch.as_tensor(plane_normal, dtype=coords.dtype, device=coords.device)
    plane_point = torch.as_tensor(plane_point, dtype=coords.dtype, device=coords.device)

    if plane_normal.shape != (3,):
        raise ValueError(f"plane_normal must have shape [3], got {list(plane_normal.shape)}.")
    if plane_point.shape != (3,):
        raise ValueError(f"plane_point must have shape [3], got {list(plane_point.shape)}.")
    normal_length = torch.linalg.vector_norm(plane_normal)
    if normal_length == 0:
        raise ValueError("Can not mirror across a plane with normal [0,0,0]. At least one dimension has to be non-zero.")

    plane_normal = plane_normal / normal_length

    distance = (coords - plane_point) @ plane_normal

    return coords - 2 * distance.unsqueeze(1) * plane_normal