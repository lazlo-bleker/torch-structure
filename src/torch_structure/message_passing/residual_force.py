import torch
from torch_geometric.nn import MessagePassing


class ResidualForce(MessagePassing):  # only works for undirected graphs!
    """Computes the per-node residual force from axial member forces and applied loads."""

    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, force, edge_index, load):
        """Sum each node's incident axial member forces and add the applied load.

        Args:
            x (torch.Tensor [N, D]): node coordinates.
            force (torch.Tensor [E]): axial force magnitude of each member.
            edge_index (torch.Tensor [2, E]): edge connectivity (undirected).
            load (torch.Tensor [N, D]): externally applied nodal loads.

        Returns:
            torch.Tensor [N, D]: the residual force at each node.
        """
        force = force.view(-1)
        out = self.propagate(edge_index, x=x, force=force)
        out = out + load
        return out

    def message(self, x_i, x_j, force):
        """Compute the force vector a member exerts on its target (center) node, along its direction."""
        vector = x_j - x_i
        vector = vector / torch.norm(vector, dim=1).unsqueeze(1) * force.unsqueeze(1)
        return vector
