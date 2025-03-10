import torch

from torch_structure.formfinding.utils import create_xy_equilibrium_space
from torch_structure.formfinding.fdm import fdm


def tna(
    coords, is_support, load, edge_index, q_target, verbose=False
):  # only works for directed graphs
    force_density = least_squares_tna(coords, is_support, edge_index, q_target, verbose)
    coords, force = fdm(
        coords,
        load,
        is_support,
        edge_index,
        force_density,
        directed=True,
        use_batching=False,
        solve_only_z=True,
    )
    return coords, force, force_density


def least_squares_tna(coords, is_support, edge_index, q_target, verbose=False):
    """
    Finds the least squares force densities with respect to target force densities that preserve x and y coordinates
    consistent with Thrust Network Analysis (TNA).

    Args:
        coords (torch.Tensor): A tensor of shape (num_nodes, 3) containing the 3D coordinates of each node. Only
            the x and y coordinates have an effect on the output.
        is_support (torch.Tensor): A boolean tensor of shape (num_nodes, 1) indicating which nodes are fixed (True) or free (False).
        edge_index (torch.Tensor): A tensor of shape (2, num_edges) containing the indices of the nodes that form each edge. Needs to be directed.
        q_target (torch.Tensor): A tensore of shape (num_edges) containing the target force density for each edge.
    """
    q_target = q_target.view(-1)

    basis_vectors = create_xy_equilibrium_space(coords, is_support, edge_index)
    vector_coefficients = torch.linalg.lstsq(basis_vectors, q_target).solution
    q = torch.mv(basis_vectors, vector_coefficients)

    if verbose:
        print(
            f"Least Squares TNA number of degrees of freedom: {basis_vectors.shape[1]}"
        )

    return q.unsqueeze(1)
