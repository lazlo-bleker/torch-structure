import torch

from torch_structure.gmp import ResidualForce

class ResidualForceLoss(torch.nn.Module):  # only works for undirected graphs!
    def __init__(self, reduction='mean'):
        super(ResidualForceLoss, self).__init__()
        self.residual_force = ResidualForce()
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, coordinates, load, support, force, edge_index):
        residual_force = self.residual_force(coordinates, force, edge_index, load)
        residual_force = torch.norm(residual_force, dim=1)
        residual_force = residual_force[~support.bool()]  # Apply mask for free/unsupported nodes
        loss = self.mse(residual_force, torch.zeros_like(residual_force))
        return loss
