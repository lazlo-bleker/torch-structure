import torch
import numpy as np
from torch_structure.message_passing import StiffnessAggregator
from .config import NUMPY_FLOAT, TORCH_FLOAT, DEVICE


def torch_to_numpy_float(x):
    """
    Converts a torch tensor to a numpy array of floating point variables
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    return x.detach().cpu().numpy().astype(NUMPY_FLOAT, copy=False)


def numpy_to_torch_float(x, device=DEVICE):
    """
    Converts a numpy array to a torch tensor of floating point variables
    """
    x = np.asarray(x, dtype=NUMPY_FLOAT)
    return torch.as_tensor(x, dtype=TORCH_FLOAT, device=device)

def scipy_jacobian(func):
    func_grad_and_value = torch.func.grad_and_value(func)

    def func_scipy(x_np, *args):
        x_torch = numpy_to_torch_float(x_np)
        grad_torch, loss_torch = func_grad_and_value(x_torch, *args)
        loss_np = torch_to_numpy_float(loss_torch)
        grad_np = torch_to_numpy_float(grad_torch)
        func_scipy.best_loss = min(loss_np, func_scipy.best_loss)
        return loss_np, grad_np

    func_scipy.best_loss = np.inf
    return func_scipy


def edge_direction(edge_index, x, return_length=False):
    row, col = edge_index
    direction = x[col] - x[row]
    length = torch.norm(direction, dim=1, keepdim=True)
    normalized_direction = direction / (length + 1e-9)

    if return_length:
        return normalized_direction, length
    else:
        return normalized_direction


def transformation_matrix(edge_index, x, length):
    """
    Create a 3D tensor containing the transformation matrices for all edges in a 3D truss.

    Parameters:
    - edge_index: Tensor of shape [2, num_edges], containing indices of connected nodes for each edge.
    - x: Tensor of shape [num_nodes, 3], containing the coordinates (x, y, z) of each node.
    - length: Tensor of shape [num_edges], containing the length of each edge.

    Returns:
    - matrix: Tensor of shape [num_edges, 2, 6], containing the transformation matrices for all edges.
    """
    # Extract coordinates of nodes connected by edges
    coords_i = x[edge_index[0]]  # Shape: [num_edges, 3]
    coords_j = x[edge_index[1]]  # Shape: [num_edges, 3]

    # Compute direction cosines (l, m, n)
    direction_vector = coords_j - coords_i  # Shape: [num_edges, 3]
    direction_cosines = direction_vector / length.view(-1, 1)  # Shape: [num_edges, 3]

    # Initialize the transformation matrix tensor
    T = torch.zeros(
        (edge_index.size(1), 2, 6), device=x.device
    )  # Shape: [num_edges, 2, 6]

    # Directly assign the direction cosines to the appropriate places
    T[:, 0, :3] = direction_cosines  # First row maps Node 1's global displacements
    T[:, 1, 3:] = direction_cosines  # Second row maps Node 2's global displacements

    return T


def effective_stiffness(edge_index, E, A, length):
    """Compute effective stiffness."""
    stiffness = E * A / length
    aggregator = StiffnessAggregator()
    K_eff = aggregator(edge_index, stiffness)
    return K_eff
