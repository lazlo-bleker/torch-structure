import torch
from torch_geometric.nn import MessagePassing


class FEM(MessagePassing):
    """Message-passing layer performing one Jacobi-style equilibrium update of a pin-jointed truss.

    Each call aggregates the axial member forces implied by the current
    displacements into a per-node residual force, then applies a diagonal
    (per-node) stiffness correction to obtain updated displacements.
    """

    def __init__(self):
        super().__init__(aggr="add")

    def forward(
        self,
        x,
        load,
        u,
        support,
        effective_stiffness,
        edge_index,
        length,
        A,
        E,
        direction,
        transformation_matrix,
    ):
        """Perform one equilibrium-update iteration.

        Args:
            x (torch.Tensor [N, 3]): node coordinates, passed through for
                propagate's argument resolution but not used by ``message``.
            load (torch.Tensor [N, 3]): externally applied nodal loads.
            u (torch.Tensor [N, 3]): current nodal displacements.
            support (torch.Tensor [N], bool): mask of supported (fixed) nodes,
                whose displacement is reset to zero after the update.
            effective_stiffness (torch.Tensor [N]): per-node diagonal
                stiffness used to convert residual force into a displacement
                correction.
            edge_index (torch.Tensor [2, E]): edge connectivity.
            length (torch.Tensor [E]): member lengths.
            A (torch.Tensor [E]): member cross-sectional areas.
            E (torch.Tensor [E]): member Young's moduli.
            direction (torch.Tensor [E, 3]): unit vector along each member.
            transformation_matrix (torch.Tensor [E, 2, 6]): local
                displacement transformation matrix for each member.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the updated nodal
            displacements ``u_new`` and the per-node residual force before
            the update.
        """
        residual_force = load + self.propagate(
            edge_index,
            x=x,
            u=u,
            length=length,
            A=A,
            E=E,
            direction=direction,
            transformation_matrix=transformation_matrix,
        )
        effective_stiffness = effective_stiffness.view(-1, 1) + 1e-9
        u_new = u + residual_force / effective_stiffness
        u_new[support] = 0
        return u_new, residual_force

    def message(self, u_i, u_j, length, A, E, direction, transformation_matrix):
        """Compute the axial member force vector contributed to the target (center) node.

        Transforms the target/source displacements into local axial
        coordinates, derives elongation, strain and stress from the member's
        length and Young's modulus, and returns the resulting force vector
        along the member direction.
        """
        length = length.view(-1)
        A = A.view(-1)
        E = E.view(-1)
        local_u = torch.matmul(
            transformation_matrix, torch.cat([u_i, u_j], dim=1).unsqueeze(2)
        ).squeeze(2)
        elongation = local_u[:, 0] - local_u[:, 1]
        strain = elongation / length
        stress = E * strain
        force = stress * A
        force_vector = -1 * direction * force.view(-1, 1)
        return force_vector


class StiffnessAggregator(MessagePassing):
    """Sums each edge's stiffness contribution onto its target node."""

    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, edge_index, stiffness):
        """Aggregate per-edge stiffness values into a per-node effective stiffness.

        Args:
            edge_index (torch.Tensor [2, E]): edge connectivity.
            stiffness (torch.Tensor [E]): per-edge stiffness values.

        Returns:
            torch.Tensor [N]: effective stiffness at each node.
        """
        return self.propagate(edge_index, stiffness=stiffness)

    def message(self, stiffness):
        """Pass each edge's stiffness value through unchanged."""
        return stiffness
