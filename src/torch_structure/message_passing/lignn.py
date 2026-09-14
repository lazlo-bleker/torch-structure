import torch
from torch.nn import MSELoss
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import SumAggregation, MaxAggregation


class LocalLogicLoss(MessagePassing):
    """Loss function for determining whether each node is connected to the correct number of trail edges"""

    def __init__(self, reduction):
        super().__init__(aggr=SumAggregation())
        self.mseloss = MSELoss(reduction=reduction)

    def forward(self, x, edge_index, y, device):
        """Compute the local trail-connectivity loss.

        Sums each node's neighbour features and compares the resulting
        trail-edge count channel against the target ``y``.

        Args:
            x (torch.Tensor [N, C]): per-node feature/probability channels.
            edge_index (torch.Tensor [2, E]): edge connectivity.
            y (torch.Tensor [N]): target trail-edge count per node.
            device: unused, kept for interface consistency.

        Returns:
            torch.Tensor: the mean squared error loss.
        """
        aggregate = self.propagate(edge_index, x=x)
        loss = self.mseloss(aggregate[:, 1], y.float())
        return loss

    def message(self, x):
        """Pass each neighbour's features through unchanged, to be summed."""
        return x


class WeightedMax(MessagePassing):
    """Takes the weighted max of all nodes in the neighborhood by multiplying node embeddings with edge weights"""

    def __init__(self):
        super().__init__(aggr=MaxAggregation())

    def forward(self, x, edge_index, weight):
        """Take the weighted elementwise max over each node's neighborhood.

        Args:
            x (torch.Tensor [N, C]): per-node features.
            edge_index (torch.Tensor [2, E]): edge connectivity.
            weight (torch.Tensor [E]): per-edge weight applied to the source
                node's features before the max.

        Returns:
            torch.Tensor [N, C]: the aggregated features.
        """
        aggregate = self.propagate(edge_index, x=x, weight=weight)
        return aggregate

    def message(self, x_i, x_j, weight):
        """Return the elementwise max of the target's features and the weighted source features."""
        return torch.max(x_i, weight.view(-1, 1) * x_j)
