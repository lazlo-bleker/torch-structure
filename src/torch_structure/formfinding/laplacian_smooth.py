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
    """Smooth free-node coordinates towards a uniform Laplacian equilibrium via explicit mean curvature flow.

    Args:
        coords (torch.Tensor [N, D]): node coordinates, updated in place for
            the free (non-fixed) nodes.
        is_fixed (torch.Tensor [N], bool): mask of nodes to keep fixed.
        edge_index (torch.Tensor [2, E]): edge connectivity.
        tolerance (float): convergence threshold on the per-iteration
            displacement norm.
        max_iter (int): maximum number of iterations.
        verbose (bool): if ``True``, print a convergence summary.
        damping_factor (float): fraction of the Laplacian step to damp;
            ``0`` is a full explicit step, values closer to ``1`` slow
            convergence.
        laplacian (Laplacian, optional): a precomputed
            [Laplacian][torch_structure.message_passing.laplace.Laplacian]
            operator; if ``None``, one is built from ``edge_index``.

    Returns:
        tuple[torch.Tensor, int, bool]: the (in-place updated) ``coords``,
        the number of iterations run, and whether the loop converged.
    """
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
