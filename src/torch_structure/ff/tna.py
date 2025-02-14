import torch

from torch_structure.ff.utils import create_xy_equilibrium_space

def least_squares_tna(coordinates, support, edge_index, q_target):  # only works for directed graphs
    """
    Finds the least squares force densities with respect to target force densities that preserve x and y coordinates
    consistent with Thrust Network Analysis (TNA).

    Args:
        coordinates (torch.Tensor): A tensor of shape (num_nodes, 3) containing the 3D coordinates of each node. Only
            the x and y coordinates have an effect on the output.
        support (torch.Tensor): A boolean tensor of shape (num_nodes, 1) indicating which nodes are fixed (True) or free (False).
        edge_index (torch.Tensor): A tensor of shape (2, num_edges) containing the indices of the nodes that form each edge.
        q_target (torch.Tensor): A tensore of shape (num_edges) containing the target force density for each edge.
    """
    basis_vectors = create_xy_equilibrium_space(coordinates, support, edge_index)
    vector_coefficients = torch.linalg.lstsq(basis_vectors, q_target).solution
    q = torch.mv(basis_vectors, vector_coefficients)

    return q
