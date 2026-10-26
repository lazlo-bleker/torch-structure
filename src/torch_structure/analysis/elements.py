import torch


# Rotational DOFs of the 12 element DOFs, per node ordered [ux, uy, uz, rx, ry, rz]
ROTATIONAL_DOFS = (3, 4, 5, 9, 10, 11)


def local_axes(coords_i, coords_j, roll=None, vertical_tol=1e-6):
    """Compute each member's length and local coordinate system.

    The local x-axis runs from node i to node j. The local z-axis lies in the vertical plane
    containing the member and points upwards; for vertical members, where that plane is
    undefined, the local z-axis is global +X. The local y-axis completes the right-handed
    system. A ``roll`` angle then rotates the local y- and z-axes about the local x-axis.

    Args:
        coords_i (torch.Tensor [E, 3]): coordinates of each member's start node.
        coords_j (torch.Tensor [E, 3]): coordinates of each member's end node.
        roll (torch.Tensor [E] or [E, 1], optional): rotation of the cross-section about the
            local x-axis, in radians. Default is no rotation.
        vertical_tol (float): members whose direction is within this tolerance of global Z are
            treated as vertical.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: member lengths ``[E]`` and rotation matrices
        ``[E, 3, 3]`` whose rows are the local x-, y- and z-axes in global coordinates.
    """
    delta = coords_j - coords_i
    length = torch.linalg.norm(delta, dim=1)
    x = delta / length.unsqueeze(1)

    global_z = torch.tensor([0.0, 0.0, 1.0], dtype=x.dtype, device=x.device).expand_as(x)
    global_x = torch.tensor([1.0, 0.0, 0.0], dtype=x.dtype, device=x.device).expand_as(x)
    is_vertical = (x[:, 2].abs() > 1 - vertical_tol).unsqueeze(1)
    reference = torch.where(is_vertical, global_x, global_z)

    y = torch.linalg.cross(reference, x)
    y = y / torch.linalg.norm(y, dim=1, keepdim=True)
    z = torch.linalg.cross(x, y)

    if roll is not None:
        roll = roll.view(-1, 1).to(x.dtype)
        cos, sin = torch.cos(roll), torch.sin(roll)
        y, z = cos * y + sin * z, -sin * y + cos * z

    return length, torch.stack([x, y, z], dim=1)


def transformation_matrix(rotation):
    """Expand per-member 3x3 rotation matrices to the 12x12 member transformation matrix.

    Args:
        rotation (torch.Tensor [E, 3, 3]): rotation matrices from
            [local_axes][torch_structure.analysis.elements.local_axes].

    Returns:
        torch.Tensor [E, 12, 12]: block-diagonal matrices mapping global to local member DOFs.
    """
    num_members = rotation.shape[0]
    T = torch.zeros(num_members, 12, 12, dtype=rotation.dtype, device=rotation.device)
    for k in range(4):
        T[:, 3 * k : 3 * k + 3, 3 * k : 3 * k + 3] = rotation
    return T


def local_stiffness(length, E, A, Iy, Iz, J, G):
    """Assemble the 12x12 Euler-Bernoulli frame stiffness matrix of each member in local coordinates.

    The DOFs per node are ordered ``[ux, uy, uz, rx, ry, rz]``, node i first. Bending about the
    local z-axis (displacement in local y) uses ``Iz``, and bending about the local y-axis
    (displacement in local z) uses ``Iy``. A member with ``Iy = Iz = J = 0`` is a truss bar.

    Args:
        length (torch.Tensor [E]): member lengths.
        E (torch.Tensor [E]): Young's moduli.
        A (torch.Tensor [E]): cross-sectional areas.
        Iy (torch.Tensor [E]): second moments of area about the local y-axis.
        Iz (torch.Tensor [E]): second moments of area about the local z-axis.
        J (torch.Tensor [E]): torsion constants.
        G (torch.Tensor [E]): shear moduli.

    Returns:
        torch.Tensor [E, 12, 12]: local stiffness matrices.
    """
    L = length
    k = torch.zeros(L.shape[0], 12, 12, dtype=L.dtype, device=L.device)

    def add(dofs, coefficients):
        # scatter a 2x2 or 4x4 block of per-member coefficients into k
        for a, row in enumerate(dofs):
            for b, col in enumerate(dofs):
                k[:, row, col] = k[:, row, col] + coefficients[a][b]

    axial = E * A / L
    add((0, 6), [[axial, -axial], [-axial, axial]])

    torsion = G * J / L
    add((3, 9), [[torsion, -torsion], [-torsion, torsion]])

    # bending in the local x-y plane: displacement uy, rotation rz
    c = E * Iz / L**3
    add(
        (1, 5, 7, 11),
        [
            [12 * c, 6 * L * c, -12 * c, 6 * L * c],
            [6 * L * c, 4 * L**2 * c, -6 * L * c, 2 * L**2 * c],
            [-12 * c, -6 * L * c, 12 * c, -6 * L * c],
            [6 * L * c, 2 * L**2 * c, -6 * L * c, 4 * L**2 * c],
        ],
    )

    # bending in the local x-z plane: displacement uz, rotation ry (opposite rotation sign)
    c = E * Iy / L**3
    add(
        (2, 4, 8, 10),
        [
            [12 * c, -6 * L * c, -12 * c, -6 * L * c],
            [-6 * L * c, 4 * L**2 * c, 6 * L * c, 2 * L**2 * c],
            [-12 * c, 6 * L * c, 12 * c, 6 * L * c],
            [-6 * L * c, 2 * L**2 * c, 6 * L * c, 4 * L**2 * c],
        ],
    )

    return k


def condense_releases(k, release):
    """Apply member end releases to local stiffness matrices by static condensation.

    Releasing torsion at either end removes the member's torsional stiffness. Released bending
    rotations are condensed out, so the member carries no moment about that axis at that end and
    the released DOFs' rows and columns become zero. Releasing every rotation at both ends turns
    the member into a pin-ended truss bar.

    Args:
        k (torch.Tensor [E, 12, 12]): local stiffness matrices.
        release (torch.Tensor [E, 6], bool): released rotations, ordered
            ``[rx_i, ry_i, rz_i, rx_j, ry_j, rz_j]`` in local member axes.

    Returns:
        torch.Tensor [E, 12, 12]: condensed local stiffness matrices.
    """
    release = release.view(-1, 6).to(torch.bool)

    # torsion: a torsion-released member carries no torque at all
    no_torsion = release[:, 0] | release[:, 3]
    keep = torch.ones(k.shape[0], 12, dtype=k.dtype, device=k.device)
    keep[:, 3] = torch.where(no_torsion, 0.0, 1.0).to(k.dtype)
    keep[:, 9] = keep[:, 3]
    k = k * keep.unsqueeze(2) * keep.unsqueeze(1)

    # bending: condense K_c = K - K M (M K M + I - M)^-1 M K, with M selecting released rotations.
    # The rotational bending block is positive definite, so the system is always invertible.
    M = torch.zeros(k.shape[0], 12, dtype=k.dtype, device=k.device)
    M[:, [4, 5, 10, 11]] = release[:, [1, 2, 4, 5]].to(k.dtype)
    if not M.any():
        return k

    M_left = M.unsqueeze(2)
    M_right = M.unsqueeze(1)
    # released DOFs without bending stiffness (e.g. Iz = 0) have zero rows; a unit diagonal keeps
    # the system solvable without coupling them to anything
    no_stiffness = (torch.diagonal(k, dim1=1, dim2=2) == 0).to(k.dtype) * M
    released_block = M_left * k * M_right + torch.diag_embed(1 - M + no_stiffness)
    X = torch.linalg.solve(released_block, M_left * k)
    k = k - k @ X

    # remove round-off left in the released rows and columns
    keep = 1 - M
    return k * keep.unsqueeze(2) * keep.unsqueeze(1)
