import torch
import scipy.linalg


def create_branch_node_matrix(edge_index):
    """
    Creates the branch-node matrix for a given set of edges.

    Args:
        edge_index (torch.Tensor): A tensor of shape (2, num_edges) containing the indices of the nodes that form each edge.

    Returns:
        torch.Tensor: A dense tensor of shape (num_edges, num_nodes) representing the branch-node incidence matrix.
            The matrix has an entry of +1 and -1 for each row indicating the start and end nodes of each edge, respectively.
    """
    device = edge_index.device
    num_edges = edge_index.size(1)
    num_nodes = torch.max(edge_index) + 1
    row_indices = torch.arange(num_edges, device=device).repeat(2)
    col_indices = edge_index.view(-1)
    indices = torch.stack([row_indices, col_indices])
    values = torch.cat(
        [torch.ones(num_edges, device=device), -torch.ones(num_edges, device=device)]
    )
    branch_node_matrix = torch.sparse_coo_tensor(
        indices, values, size=(num_edges, num_nodes)
    ).to_dense()
    return branch_node_matrix


def create_xy_equilibrium_space(coords, is_support, edge_index):
    """Compute a basis for the space of xy-equilibrium-preserving coordinate perturbations.

    Builds the xy equilibrium matrix and returns a basis of its null space,
    i.e. the directions in which free-node xy-coordinates can move while
    keeping the horizontal equilibrium equations satisfied.

    Args:
        coords (torch.Tensor [N, 3]): node coordinates; only x and y affect
            the result.
        is_support (torch.Tensor [N, 1], bool): mask of fixed (supported)
            nodes.
        edge_index (torch.Tensor [2, E]): directed edge connectivity.

    Returns:
        torch.Tensor [D, K]: basis vectors (columns) spanning the null space
        of the xy equilibrium matrix, where ``D`` is twice the number of
        free nodes.
    """
    coords = torch.clone(coords)  # check if this is necessary

    A = create_xy_equilibrium_matrix(coords, is_support, edge_index)

    basis_vectors = scipy.linalg.null_space(A.cpu().numpy())  # consider using torch
    basis_vectors = torch.tensor(basis_vectors, dtype=torch.float32)

    return basis_vectors


def create_xy_equilibrium_matrix(coords, is_support, edge_index):
    """
    Creates the equilibrium matrix for the x and y components of the equilibrium equations.

    Args:
        coords (torch.Tensor): A tensor of shape (num_nodes, 3) containing the 3D coordinates of each node. Only
            the x and y coordinates have an effect on the output.
        is_support (torch.Tensor): A boolean tensor of shape (num_nodes, 1) indicating which nodes are fixed (True) or free (False).
        edge_index (torch.Tensor): A tensor of shape (2, num_edges) containing the indices of the nodes that form each edge. Needs to be directed.
    """
    is_support = is_support.view(-1)

    coords = torch.clone(coords)  # check if this is necessary

    C = create_branch_node_matrix(edge_index)
    C_free = C[:, ~is_support]
    C_free_transposed = torch.transpose(C_free, 0, 1)

    u = torch.mv(C, coords[:, 0])
    v = torch.mv(C, coords[:, 1])
    U = torch.diag(u)
    V = torch.diag(v)
    A = torch.vstack((torch.mm(C_free_transposed, U), torch.mm(C_free_transposed, V)))

    return A
