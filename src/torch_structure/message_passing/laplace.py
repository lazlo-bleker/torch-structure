from torch_geometric.nn import MessagePassing
import torch


class Laplacian(MessagePassing):
    """Matrix-free uniform graph Laplacian.

    The operator uses the random-walk form of the uniform Laplacian:
    ``Lx = x - mean(neighbour_values)``.
    """

    def __init__(self, edge_index, num_nodes=None):
        super().__init__(aggr="mean")
        self.register_buffer("edge_index", edge_index)
        self.num_nodes = num_nodes

    def forward(self, x):
        mean_neighbours = self.propagate(self.edge_index, x=x, size=(self.num_nodes, self.num_nodes))
        return x - mean_neighbours

    def dense_matrix(self):
        """Materialize the operator as a dense matrix when a solver needs it."""
        identity = torch.eye(
            self.num_nodes, device=self.edge_index.device, dtype=torch.get_default_dtype()
        )
        return self(identity)

    def message(self, x_j):
        return x_j
