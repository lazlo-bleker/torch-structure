import torch
from torch.nn import NLLLoss
from torch_structure.message_passing import LocalLogicLoss, WeightedMax


class LogicLoss(torch.nn.Module):
    """Theory-informed loss function for determining topology diagram validity"""

    def __init__(self, search_depth, device, reduction="mean"):
        super(LogicLoss, self).__init__()
        self.search_depth = search_depth
        self.device = device
        self.local = LocalLogicLoss(reduction=reduction)
        self.wmax = WeightedMax()
        self.nllloss = NLLLoss(reduction=reduction)
        self.mean_dim = (0, 1) if reduction == "mean" else 1

    def forward(self, x, edge_index, batch, y, x2):
        # Compress domain to prevent numerical issues with ones and zeros
        x = x * 0.9999999 + 0.00000005

        # Compute local loss related to local trail connectivity
        local_loss = self.local(x, edge_index, y, self.device)

        # Compute trail loss related to indirect connectivity of nodes to support/origin nodes through trail edges
        for i in range(self.search_depth):
            x2 = self.wmax(x2, edge_index, weight=x[:, 1])
        trail_loss = torch.mean(-torch.log(x2), dim=self.mean_dim)

        # Compute weighted average according to local_component
        local_component = 0.9
        loss = local_component * local_loss + (1 - local_component) * trail_loss
        return loss, local_loss, trail_loss
