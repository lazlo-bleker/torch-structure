import torch

from torch_structure.message_passing.laplacian_smooth import LaplacianSmoothing

def laplacian_smoothing(coords, is_fixed, edge_index, tolerance=1e-7, max_iter=10000, verbose=False):
    is_fixed = is_fixed.view(-1)

    laplace_update = LaplacianSmoothing(damping_factor=0.5)
    converged = False
    for i in range(max_iter):
        updated_coords = laplace_update(coords, edge_index)[~is_fixed]
        delta = coords[~is_fixed] - updated_coords
        delta_norm = torch.norm(delta)
        coords[~is_fixed] = updated_coords
        if delta_norm < tolerance:
            converged = True
            break

    if verbose:
        print(f"Laplacian Smoothing finished in {i} iterations. Converged: {converged}.")

    return coords, i, converged
