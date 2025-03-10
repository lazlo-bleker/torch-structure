import torch
from torch.nn import MSELoss
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import SumAggregation, MaxAggregation

class LocalLogicLoss(MessagePassing):
    """ Loss function for determining whether each node is connected to the correct number of trail edges """
    def __init__(self, reduction):
        super().__init__(aggr=SumAggregation())
        self.mseloss = MSELoss(reduction=reduction)

    def forward(self, x, edge_index, y, device):
        aggregate = self.propagate(edge_index, x=x)
        loss = self.mseloss(aggregate[:, 1], y.float())
        return loss

    def message(self, x):
        return x

class WeightedMax(MessagePassing):
    """ Takes the weighted max of all nodes in the neighborhood by multiplying node embeddings with edge weights """
    def __init__(self):
        super().__init__(aggr=MaxAggregation())

    def forward(self, x, edge_index, weight):
        aggregate = self.propagate(edge_index, x=x, weight=weight)
        return aggregate

    def message(self, x_i, x_j, weight):
        return torch.max(x_i, weight.view(-1, 1) * x_j)
