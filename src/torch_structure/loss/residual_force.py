import torch

from torch_structure.message_passing import ResidualForce

class ResidualForceLoss(torch.nn.Module):  # only works for undirected graphs!
    def __init__(self, reduction='mean'):
        super(ResidualForceLoss, self).__init__()
        self.residual_force = ResidualForce()
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, coords, load, is_support, force, edge_index):
        is_support = is_support.view(-1)
        residual_force = self.residual_force(coords, force, edge_index, load)
        residual_force = torch.norm(residual_force, dim=1)
        residual_force = residual_force[~is_support]  # Apply mask for free/unsupported nodes
        loss = self.mse(residual_force, torch.zeros_like(residual_force))
        return loss
