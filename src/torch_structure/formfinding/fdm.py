import torch

from torch_structure.formfinding.utils import create_branch_node_matrix


def fdm(
    coords,
    load,
    is_support,
    edge_index,
    force_density,
    batch=None,
    use_batching=True,
    directed=False,
    solve_only_z=False,
    C=None,
):
    """
    Performs the Force Density Method (FDM) to find the equilibrium state of a structure given its coordinates, loads,
    supports, edge connections, and force densities.

    Args:
        coords (torch.Tensor): A tensor of shape (num_nodes, 3) containing the initial 3D coordinates of each node. Only
            coordinates of supported nodes have an effect on the output.
        load (torch.Tensor): A tensor of shape (num_nodes, 3) containing the external load vector for each node.
        is_support (torch.Tensor): A boolean tensor of shape (num_nodes, 1) indicating which nodes are fixed (True) or free (False).
        edge_index (torch.Tensor): A tensor of shape (2, num_edges) containing the indices of the nodes that form each edge.
        force_density (torch.Tensor): A tensor of shape (num_edges, 1) containing the force density for each edge.
        batch (torch.Tensor, optional): A tensor of shape (num_nodes,) assigning each node to a graph of the batch,
            sorted as in a PyTorch Geometric batch. If supplied, the linear system of every graph is assembled as a
            separate block, padded to the size of the largest graph. If not supplied, a single linear system
            containing all nodes and edges is solved. Default is None.
        use_batching (bool, optional): If True, the blocks of all graphs are solved at once with one batched solve,
            otherwise one after another. Only used with ``batch``. Default is True.
        directed (bool, optional): If True, edges are assumed to be directed. If False, the force densities are halved to account for
            undirected edges. Default is False.
        solve_only_z (bool, optional): If True, only the z-coordinate of the free nodes is solved. Default is False.
        C (torch.Tensor, optional): A precomputed branch-node matrix. If None, it is created using the edge_index. Not
            used with ``batch``. Default is None.

    Returns:
        A tuple containing:
            - coords (torch.Tensor): A tensor of shape (num_nodes, 3) containing form-found coordinates of each node.
            - force (torch.Tensor): A tensor of shape (num_edges, 1) containing the axial force of each edge.
    """
    is_support = is_support.view(-1)
    force_density = force_density.view(-1)

    coords = torch.clone(coords)  # check if this is necessary

    # halve force densities to account for undirected graph
    if not directed:
        force_density = 0.5 * force_density

    if batch is not None:
        coords = _fdm_batched(coords, load, is_support, edge_index, force_density, batch, use_batching, solve_only_z)
        src, dst = edge_index
        lengths = torch.norm(coords[src] - coords[dst], dim=1)
    else:
        if C is None:
            C = create_branch_node_matrix(edge_index, dtype=coords.dtype)
        C_free = C[:, ~is_support]
        C_fixed = C[:, is_support]
        C_free_transposed = torch.transpose(C_free, 0, 1)

        Q = torch.diag(force_density)
        A = torch.mm(torch.mm(C_free_transposed, Q), C_free)  # >1/3 of computation time

        D_f = torch.mm(torch.mm(C_free_transposed, Q), C_fixed)
        if solve_only_z:
            b_1 = load[~is_support, 2]
            b_2 = torch.mv(D_f, coords[is_support, 2])
        else:
            b_1 = load[~is_support]
            b_2 = torch.mm(D_f, coords[is_support])
        b = b_1 - b_2

        form_found_coords = torch.linalg.solve(A, b)

        if solve_only_z:
            coords[~is_support, 2] = form_found_coords
        else:
            coords[~is_support, :] = form_found_coords
        lengths = torch.norm(torch.mm(C, coords), dim=1)
    force = force_density * lengths

    # double force to account for undirected graph
    if not directed:
        force = 2 * force

    return coords, force.unsqueeze(1)


def _fdm_batched(coords, load, is_support, edge_index, force_density, batch, use_batching, solve_only_z):
    """
    Solves the FDM for every graph of a batch, assembling each graph's linear system as its own block.

    Every block has one row and column per node of its graph, padded to the size of the largest graph. Rows of
    supported and padding nodes are identity rows whose solution is the node's own coordinates (zero for padding),
    so the free nodes of every block solve ``A_free x_free = load_free - D_fixed x_fixed``, as without a batch.

    Args:
        coords (torch.Tensor [N, 3]): node coordinates; only those of supported nodes are used.
        load (torch.Tensor [N, 3]): node loads.
        is_support (torch.Tensor [N], bool): supported nodes.
        edge_index (torch.Tensor [2, E]): edges, which never connect two graphs.
        force_density (torch.Tensor [E]): force density of every edge, already halved for undirected edges.
        batch (torch.Tensor [N]): graph of every node, sorted.
        use_batching (bool): solve all blocks at once instead of one after another.
        solve_only_z (bool): solve only the z-coordinates of the free nodes.

    Returns:
        torch.Tensor [N, 3]: form-found coordinates.
    """
    batch = batch.view(-1)
    dtype, device = coords.dtype, coords.device
    num_graphs = int(batch.max()) + 1
    node_counts = torch.bincount(batch, minlength=num_graphs)
    ptr = torch.cat([node_counts.new_zeros(1), torch.cumsum(node_counts, dim=0)])
    size = int(node_counts.max())
    local = torch.arange(batch.size(0), device=device) - ptr[batch]  # index of every node in its own graph
    dims = [2] if solve_only_z else [0, 1, 2]

    # the force density matrix C^T Q C of every graph, summed edge by edge without building C
    src, dst = edge_index
    graph = batch[src]
    L = torch.zeros((num_graphs, size, size), dtype=dtype, device=device)
    q = force_density.to(dtype)
    L = L.index_put((graph, local[src], local[src]), q, accumulate=True)
    L = L.index_put((graph, local[dst], local[dst]), q, accumulate=True)
    L = L.index_put((graph, local[src], local[dst]), -q, accumulate=True)
    L = L.index_put((graph, local[dst], local[src]), -q, accumulate=True)

    # padding nodes count as supported, at the origin
    fixed = torch.ones((num_graphs, size), dtype=torch.bool, device=device)
    fixed[batch, local] = is_support
    padded_coords = torch.zeros((num_graphs, size, len(dims)), dtype=dtype, device=device)
    padded_coords = padded_coords.index_put((batch, local), coords[:, dims])
    padded_load = torch.zeros((num_graphs, size, len(dims)), dtype=dtype, device=device)
    padded_load = padded_load.index_put((batch, local), load[:, dims].to(dtype))

    free = ~fixed
    free_rows, free_cols = free.unsqueeze(2), free.unsqueeze(1)
    A = L * (free_rows & free_cols) + torch.diag_embed(fixed.to(dtype))
    D_fixed = L * (free_rows & ~free_cols)
    b = torch.where(free_rows, padded_load - torch.bmm(D_fixed, padded_coords), padded_coords)

    if use_batching:
        solution = torch.linalg.solve(A, b)
    else:
        solution = torch.stack([torch.linalg.solve(A[i], b[i]) for i in range(num_graphs)])

    coords = torch.clone(coords)
    coords[:, dims] = solution[batch, local]
    return coords
