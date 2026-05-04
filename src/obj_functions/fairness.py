from torch_structure.data import StructData
import torch
from torch_geometric.nn import MessagePassing
from torch_structure.message_passing.laplacian_smooth import LaplacianSmoothing

def fairness_cache(graph_solved: "StructData"):
    kwargs = {}
    kwargs["laplace"] = LaplacianSmoothing()
    # Detect boundary edges
    is_boundary = graph_solved.is_support + (graph_solved.sequence == 0)
    kwargs["is_boundary"] = is_boundary.squeeze(1)
    return kwargs

def _fairness_func(graph_solved: "StructData", laplace : MessagePassing, is_boundary):
    mean_neighbours = laplace.propagate(graph_solved.edge_index, x=graph_solved.coords)
    laplacian = graph_solved.coords - mean_neighbours
    masked_laplacian = laplacian[~is_boundary]
    return masked_laplacian

def fairness_func(*args, **kwargs):
    masked_laplacian = _fairness_func(*args, **kwargs)
    return torch.sum(masked_laplacian ** 2)

def graph_post_process(graph):
    kwargs = fairness_cache(graph_solved=graph)
    laplacian = torch.zeros_like(graph.coords)
    masked_laplacian = _fairness_func(graph, **kwargs)
    laplacian[~kwargs["is_boundary"]] = masked_laplacian
    setattr(graph, "laplacian", laplacian)
    return graph