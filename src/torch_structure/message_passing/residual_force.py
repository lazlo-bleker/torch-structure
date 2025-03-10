import torch
from torch_geometric.nn import MessagePassing


class ResidualForce(MessagePassing):  # only works for undirected graphs!
    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, force, edge_index, load):
        force = force.view(-1)
        out = self.propagate(edge_index, x=x, force=force)
        out = out + load
        return out

    def message(self, x_i, x_j, force):
        vector = x_j - x_i
        vector = vector / torch.norm(vector, dim=1).unsqueeze(1) * force.unsqueeze(1)
        return vector
