import torch
import numpy as np
from torch_structure.message_passing import StiffnessAggregator


def scipy_jacobian(func):
    """Wrap a scalar-valued torch function for use with scipy's gradient-based optimizers.

    Args:
        func (Callable[..., torch.Tensor]): differentiable function that
            takes a 1D torch tensor (plus optional extra args) and returns a
            scalar loss.

    Returns:
        Callable[[numpy.ndarray, ...], tuple[float, numpy.ndarray]]: a
        function taking a numpy array (plus the same extra args) that
        returns the ``(loss, gradient)`` pair expected by
        ``scipy.optimize.minimize(jac=True)``. The returned function tracks
        the best loss seen so far in its ``best_loss`` attribute.
    """
    func_grad_and_value = torch.func.grad_and_value(func)

    def func_scipy(x_np, *args):
        """Evaluate ``func`` and its gradient at ``x_np``, tracking the best loss seen."""
        x = torch.tensor(x_np, dtype=torch.float64)
        grad_val, loss_val = func_grad_and_value(x, *args)
        loss, grad = loss_val.item(), grad_val.detach().numpy()
        func_scipy.best_loss = min(loss, func_scipy.best_loss)
        return loss, grad

    func_scipy.best_loss = np.inf
    return func_scipy


def edge_direction(edge_index, x, return_length=False):
    """Compute the unit vector (and optionally the length) along each edge.

    Args:
        edge_index (torch.Tensor [2, E]): edge connectivity.
        x (torch.Tensor [N, D]): node coordinates.
        return_length (bool): if ``True``, also return each edge's length.

    Returns:
        torch.Tensor [E, D] or tuple[torch.Tensor, torch.Tensor]: the
        normalized direction from source to target node, and, if
        ``return_length`` is ``True``, a ``[E, 1]`` tensor of edge lengths.
    """
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
