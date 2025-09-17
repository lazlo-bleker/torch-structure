import torch

def line_length(start_coords, end_coords):
    r""" Compute the length of lines defined by start and end coordinates.

    Args:
        start_coords (torch.Tensor): Tensor of shape [N, 3] representing the start coordinates of N lines.
        end_coords (torch.Tensor): Tensor of shape [N, 3] representing the end coordinates of N lines.
    
    Returns:
        (torch.Tensor): Tensor of shape [N, 1] representing the lengths of the lines.
    """
    length = torch.norm(end_coords - start_coords, dim=1, keepdim=True)
    return length

def graph_edge_lengths(coords, edge_index):
    r""" Compute the lengths of edges in a graph.

    Args:
        coords (torch.Tensor): Tensor of shape [N, 3] representing the coordinates of N nodes.
        edge_index (torch.Tensor): Tensor of shape [2, E] representing the edges in the graph,
            where each column defines an edge from edge_index[0, i] to edge_index[1, i].

    Returns:
        (torch.Tensor): Tensor of shape [E, 1] representing the lengths of the edges.
    """
    start_coords = coords[edge_index[0]]
    end_coords = coords[edge_index[1]]
    lengths = line_length(start_coords, end_coords)
    return lengths 