from torch_geometric.nn import MessagePassing

class LaplacianSmoothing(MessagePassing):
    def __init__(self, damping_factor=0.5):
        super().__init__(aggr='mean')
        self.damping_factor = damping_factor
    
    def forward(self, x, edge_index):
        mean_neighbours = self.propagate(edge_index, x=x)
        out = x + (1 - self.damping_factor) * (mean_neighbours - x)
        return out
    
    def message(self, x_i, x_j):  # is x_i necessary?
        return x_j
