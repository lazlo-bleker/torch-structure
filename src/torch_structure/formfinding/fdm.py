import torch

from torch_structure.formfinding.utils import create_branch_node_matrix

def fdm(coords, load, support, edge_index, force_density, batch=None, use_batching=True, directed=False, solve_only_z=False):
    """
    Performs the Force Density Method (FDM) to find the equilibrium state of a structure given its coordinates, loads, 
    supports, edge connections, and force densities.

    Args:
        coords (torch.Tensor): A tensor of shape (num_nodes, 3) containing the initial 3D coordinates of each node. Only
            coordinates of supported nodes have an effect on the output.
        load (torch.Tensor): A tensor of shape (num_nodes, 3) containing the external load vector for each node.
        support (torch.Tensor): A boolean tensor of shape (num_nodes, 1) indicating which nodes are fixed (True) or free (False).
        edge_index (torch.Tensor): A tensor of shape (2, num_edges) containing the indices of the nodes that form each edge.
        force_density (torch.Tensor): A tensor of shape (num_edges, 1) containing the force density for each edge.
        batch (torch.Tensor, optional): A tensor of shape (num_nodes, 1) indicating the batch assignment for each node. If not supplied
            a single linear system containing all nodes and edges of the entire batch is solved. Default is None.
        use_batching (bool, optional): If True, batching is used for solving the linear system. Default is True.

    Returns:
        A tuple containing:
            - coords (torch.Tensor): A tensor of shape (num_nodes, 3) containing form-found coordinates of each node.
            - force (torch.Tensor): A tensor of shape (num_edges, 1) containing the axial force of each edge.
    """
    support = support.view(-1)
    force_density = force_density.view(-1)

    device = coords.device
    coords = torch.clone(coords)  # check if this is necessary
    if not directed:
        force_density = 0.5*force_density  # halve force densities to account for undirected graph

    C = create_branch_node_matrix(edge_index)
    C_free = C[:, ~support]
    C_fixed = C[:, support]
    C_free_transposed = torch.transpose(C_free, 0, 1)

    Q = torch.diag(force_density)
    A = torch.mm(torch.mm(C_free_transposed, Q), C_free)  # >1/3 of computation time

    if solve_only_z:
        b_1 = load[~support, 2]
        b_2 = torch.mv(torch.mm(torch.mm(C_free_transposed, Q), C_fixed), coords[support, 2])
    else:
        b_1 = load[~support]
        b_2 = torch.mm(torch.mm(torch.mm(C_free_transposed, Q), C_fixed), coords[support])
    b = b_1 - b_2

    if batch is None:
        form_found_coords = torch.linalg.solve(A, b)
    else:
        batch_free = batch[~support]
        num_graphs = batch.max().item() + 1
        node_counts = torch.bincount(batch)
        max_size = node_counts.max().item()

        A_batched = torch.eye(max_size, device=device).unsqueeze(0).repeat(num_graphs, 1, 1)
        b_batched = torch.zeros(num_graphs, max_size, 3, device=device)
        pad_mask = torch.zeros(num_graphs * max_size, dtype=torch.bool, device=device)
        ff_coords = []
        for i in range(num_graphs):
            mask = batch_free == i

            indices = torch.nonzero(mask).squeeze()
            submatrix = A[indices][:, indices]
            A_batched[i, :submatrix.size(0), :submatrix.size(1)] = submatrix

            subvector = b[mask]
            graph_size = subvector.size(0)
            b_batched[i, :graph_size] = subvector

            pad_mask[i*max_size:i*max_size + graph_size] = 1

            if not use_batching:
                ff_coords.append(torch.linalg.solve(submatrix, subvector))
                # ff_coords.append(torch.linalg.solve(A_batched[i], b_batched[i]))

        if use_batching:
            form_found_coords = torch.linalg.solve(A_batched, b_batched)
            form_found_coords = form_found_coords.reshape(num_graphs*max_size, 3)[pad_mask]
        else:
            form_found_coords = torch.cat(ff_coords, dim=0)
            # form_found_coords = form_found_coords.reshape(num_graphs*max_size, 3)[pad_mask]

    if solve_only_z:
        coords[~support, 2] = form_found_coords
    else:
        coords[~support, :] = form_found_coords

    lengths = torch.norm(torch.mm(C, coords), dim=1)
    force = force_density * lengths
    if not directed:
        force = 2*force  # double force to account for undirected graph

    return coords, force.unsqueeze(1)