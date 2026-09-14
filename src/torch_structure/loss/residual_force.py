import torch

from torch_structure.message_passing import ResidualForce


class ResidualForceLoss(torch.nn.Module):  # only works for undirected graphs!
    """Penalizes non-zero residual force at unsupported nodes, i.e. equilibrium violation."""

    def __init__(self, reduction="mean"):
        super(ResidualForceLoss, self).__init__()
        self.residual_force = ResidualForce()
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, coords, load, is_support, force, edge_index):
        """Compute the mean squared residual force over unsupported nodes.

        Args:
            coords (torch.Tensor [N, D]): node coordinates.
            load (torch.Tensor [N, D]): externally applied nodal loads.
            is_support (torch.Tensor [N], bool): mask of supported (fixed) nodes,
                excluded from the loss since they may carry a reaction force.
            force (torch.Tensor [E]): axial force magnitude of each member.
            edge_index (torch.Tensor [2, E]): edge connectivity (undirected).

        Returns:
            torch.Tensor: the mean squared residual force.
        """
        is_support = is_support.view(-1)
        residual_force = self.residual_force(coords, force, edge_index, load)
        residual_force = torch.norm(residual_force, dim=1)

        # Apply mask for free/unsupported nodes
        residual_force = residual_force[~is_support]

        loss = self.mse(residual_force, torch.zeros_like(residual_force))
        return loss
