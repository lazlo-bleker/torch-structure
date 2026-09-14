import torch

from torch_structure.message_passing.laplace import Laplacian


def laplacian_smoothing(
    coords,
    is_fixed,
    edge_index,
    tolerance=1e-7,
    max_iter=10000,
    verbose=False,
    damping_factor = 0.5,
    laplacian=None,
):
    is_fixed = is_fixed.view(-1)

    if laplacian is None:
        laplacian = Laplacian(edge_index, num_nodes=coords.size(0))
    
    converged = False
    for i in range(max_iter):
        # Explicit time-stepping associated with mean curvature flow
        updated_coords = coords - (1 - damping_factor) * laplacian(coords)
        updated_coords = updated_coords[~is_fixed]
        delta = coords[~is_fixed] - updated_coords
        delta_norm = torch.norm(delta)
        coords[~is_fixed] = updated_coords
        if delta_norm < tolerance:
            converged = True
            break

    if verbose:
        print(
            f"Laplacian Smoothing finished in {i} iterations. Converged: {converged}."
        )

    return coords, i, converged
