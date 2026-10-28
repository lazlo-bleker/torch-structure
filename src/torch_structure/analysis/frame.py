from typing import NamedTuple

import torch

from torch_structure.analysis.elements import (
    condense_releases,
    local_axes,
    local_stiffness,
    transformation_matrix,
)


class FrameResult(NamedTuple):
    """Results of [frame_analysis][torch_structure.analysis.frame.frame_analysis].

    Attributes:
        displacement (torch.Tensor [N, 3]): nodal translations.
        rotation (torch.Tensor [N, 3]): nodal rotations about the global axes.
        force (torch.Tensor [E, 1]): axial member forces, tension positive.
        end_forces (torch.Tensor [E, 12]): member end forces in local member axes, ordered
            ``[N, Vy, Vz, T, My, Mz]`` at node i followed by node j, as exerted by the nodes on
            the member.
        reaction_force (torch.Tensor [N, 3]): support reaction forces, zero at free DOFs.
        reaction_moment (torch.Tensor [N, 3]): support reaction moments about the global axes,
            zero at free DOFs.
    """

    displacement: torch.Tensor
    rotation: torch.Tensor
    force: torch.Tensor
    end_forces: torch.Tensor
    reaction_force: torch.Tensor
    reaction_moment: torch.Tensor


def frame_analysis(
    coords,
    edge_index,
    load,
    support_dofs,
    E,
    A,
    Iy=None,
    Iz=None,
    J=None,
    G=None,
    roll=None,
    release=None,
    moment=None,
    batch=None,
    dtype=torch.float64,
    null_tol=1e-9,
):
    """
    Performs a linear-elastic analysis of a 3D frame of Euler-Bernoulli members, which may be
    mixed with pin-ended truss bars.

    Every node has six DOFs ``[ux, uy, uz, rx, ry, rz]``. A member carries axial force, torsion
    and bending, minus whatever its ``release`` frees: releasing every rotation at both ends
    gives a pin-ended truss bar, as does leaving ``Iy``, ``Iz`` and ``J`` at zero. Nodal rotations
    that no member restrains, such as at the joints of a pure truss, are fixed automatically, so
    trusses need no rotational supports.

    Args:
        coords (torch.Tensor): A tensor of shape (num_nodes, 3) containing the coordinates of
            each node.
        edge_index (torch.Tensor): A tensor of shape (2, num_members) with the start and end node
            of each member.
        load (torch.Tensor): A tensor of shape (num_nodes, 3) containing the nodal forces.
        support_dofs (torch.Tensor): A boolean tensor of shape (num_nodes, 6) marking restrained DOFs
            ``[ux, uy, uz, rx, ry, rz]``, e.g.
            [StructData.support_dofs][torch_structure.data.data.StructData.support_dofs].
        E (torch.Tensor or float): Young's modulus per member, shape (num_members, 1).
        A (torch.Tensor or float): Cross-sectional area per member, shape (num_members, 1).
        Iy (torch.Tensor or float, optional): Second moment of area about the local y-axis per
            member. Default is 0.
        Iz (torch.Tensor or float, optional): Second moment of area about the local z-axis per
            member. Default is 0.
        J (torch.Tensor or float, optional): Torsion constant per member. Default is 0.
        G (torch.Tensor or float, optional): Shear modulus per member. Default is ``E / 2.6``
            (Poisson's ratio 0.3).
        roll (torch.Tensor or float, optional): Rotation of each member's cross-section about its
            axis, in radians. See
            [local_axes][torch_structure.analysis.elements.local_axes]. Default is 0.
        release (torch.Tensor, optional): A boolean tensor of shape (num_members, 6) marking
            released end rotations ``[rx_i, ry_i, rz_i, rx_j, ry_j, rz_j]`` in local member axes.
            Default is no releases (rigid joints).
        moment (torch.Tensor, optional): A tensor of shape (num_nodes, 3) containing nodal
            moments about the global axes. Default is no moments.
        batch (torch.Tensor, optional): A tensor of shape (num_nodes,) assigning each node to a
            graph. Default is a single graph.
        dtype (torch.dtype, optional): Precision of the assembly and solve. Results are cast back
            to the dtype of ``coords``. Default is ``torch.float64``.
        null_tol (float, optional): Relative tolerance below which a nodal rotation counts as
            unrestrained by any member. Default is 1e-9.

    Returns:
        FrameResult: A named tuple of ``displacement``, ``rotation``, ``force``, ``end_forces``,
        ``reaction_force`` and ``reaction_moment``; see [FrameResult][torch_structure.analysis.frame.FrameResult].
    """
    out_dtype = coords.dtype
    device = coords.device
    num_nodes = coords.shape[0]
    num_members = edge_index.shape[1]
    src, dst = edge_index[0], edge_index[1]

    def member_attr(value, default):
        if value is None:
            value = default
        value = torch.as_tensor(value, device=device).to(dtype)
        return value.expand(num_members) if value.numel() == 1 else value.reshape(-1)

    E = member_attr(E, None)
    A = member_attr(A, None)
    Iy = member_attr(Iy, 0.0)
    Iz = member_attr(Iz, 0.0)
    J = member_attr(J, 0.0)
    G = member_attr(G, E / 2.6)
    roll = None if roll is None else member_attr(roll, 0.0)

    # Member stiffness in global coordinates
    coords = coords.to(dtype)
    length, rotation = local_axes(coords[src], coords[dst], roll)
    k_local = local_stiffness(length, E, A, Iy, Iz, J, G)
    if release is not None:
        k_local = condense_releases(k_local, release)
    T = transformation_matrix(rotation)
    k_global = T.transpose(1, 2) @ k_local @ T

    # DOF numbering, padded per graph
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    batch = batch.view(-1)
    num_graphs = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=num_graphs)
    local_node = torch.arange(num_nodes, device=device) - (torch.cumsum(counts, 0) - counts)[batch]
    num_dofs = 6 * int(counts.max().item())

    node_dof = 6 * local_node.unsqueeze(1) + torch.arange(6, device=device)  # [N, 6]
    node_flat = batch.unsqueeze(1) * num_dofs + node_dof  # [N, 6] into [G * num_dofs]

    # Assembly
    member_dof = torch.cat([node_dof[src], node_dof[dst]], dim=1)  # [E, 12]
    member_offset = (batch[src] * num_dofs * num_dofs).view(-1, 1, 1)
    member_flat = member_offset + member_dof.unsqueeze(2) * num_dofs + member_dof.unsqueeze(1)
    K = torch.zeros(num_graphs * num_dofs * num_dofs, dtype=dtype, device=device)
    K = K.index_add(0, member_flat.reshape(-1), k_global.reshape(-1))
    K = K.view(num_graphs, num_dofs, num_dofs)

    if moment is None:
        moment = torch.zeros_like(load)
    nodal_load = torch.cat([load, moment], dim=1).to(dtype)
    F = torch.zeros(num_graphs * num_dofs, dtype=dtype, device=device)
    F = F.index_add(0, node_flat.reshape(-1), nodal_load.reshape(-1)).view(num_graphs, num_dofs)

    # Restrained DOFs, including padding, are removed by a unit-scaled diagonal
    support_dofs = support_dofs.to(device=device, dtype=torch.bool)
    if support_dofs.shape != (num_nodes, 6):
        raise ValueError(f"support_dofs must have shape ({num_nodes}, 6), but got {tuple(support_dofs.shape)}.")
    fixed = torch.ones(num_graphs * num_dofs, dtype=torch.bool, device=device)
    fixed[node_flat.reshape(-1)] = support_dofs.reshape(-1)
    fixed = fixed.view(num_graphs, num_dofs)
    free = (~fixed).to(dtype)

    scale = torch.diagonal(K, dim1=1, dim2=2).detach().abs().amax(dim=1).clamp_min(1.0)
    K_reduced = K * free.unsqueeze(2) * free.unsqueeze(1) + torch.diag_embed(fixed * scale.view(-1, 1))
    F_reduced = F * free

    # Nodal rotations no member restrains are null directions of the stiffness matrix. A spring
    # along exactly these directions makes the system solvable without changing the solution.
    rot_flat = node_dof[:, 3:]
    rows = rot_flat.unsqueeze(2).expand(-1, 3, 3)
    cols = rot_flat.unsqueeze(1).expand(-1, 3, 3)
    graph = batch.view(-1, 1, 1).expand(-1, 3, 3)
    with torch.no_grad():
        eigenvalues, eigenvectors = torch.linalg.eigh(K_reduced[graph, rows, cols])
        node_scale = scale[batch].view(-1, 1)
        is_null = (eigenvalues <= null_tol * node_scale).to(dtype)
        spring = eigenvectors @ torch.diag_embed(is_null * node_scale) @ eigenvectors.transpose(1, 2)
    K_reduced = K_reduced.index_put((graph, rows, cols), spring, accumulate=True)

    # Solve
    u = torch.linalg.solve(K_reduced, F_reduced.unsqueeze(2)).squeeze(2)
    u_nodes = u.reshape(-1)[node_flat]  # [N, 6]

    reaction = (K @ u.unsqueeze(2)).squeeze(2) - F
    reaction = reaction.reshape(-1)[node_flat] * support_dofs.to(dtype)

    # Member end forces in local axes
    u_members = torch.cat([u_nodes[src], u_nodes[dst]], dim=1).unsqueeze(2)
    end_forces = (k_local @ (T @ u_members)).squeeze(2)
    force = end_forces[:, 6:7]

    return FrameResult(
        displacement=u_nodes[:, :3].to(out_dtype),
        rotation=u_nodes[:, 3:].to(out_dtype),
        force=force.to(out_dtype),
        end_forces=end_forces.to(out_dtype),
        reaction_force=reaction[:, :3].to(out_dtype),
        reaction_moment=reaction[:, 3:].to(out_dtype),
    )
