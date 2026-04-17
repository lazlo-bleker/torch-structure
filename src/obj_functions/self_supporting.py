import os
import imageio
import shutil
from torch_structure.data import StructData
import torch
from config import TORCH_FLOAT
import matplotlib.pyplot as plt


def supporting_loss_cache(graph_solved: "StructData"):
    kwargs = {}
    # Validate that the assembly sequence starts at 0
    graph_solved.assembly_sequence -= int(min(graph_solved.assembly_sequence))
    kwargs["n_nodes_full"] = graph_solved.num_nodes
    kwargs["n_edges_full"] = graph_solved.num_edges // 2  # Account for a directed graph
    n_steps = int(max(graph_solved.assembly_sequence)) + 1
    kwargs["steps"] = torch.arange(n_steps)

    # Cache indices to populate load bearing matrix
    head = graph_solved.edge_index[1]
    offsets = torch.tensor(
        [0, graph_solved.num_nodes, 2 * graph_solved.num_nodes],
        device=head.device,
    )
    edge_apply_node = head.unsqueeze(0) + offsets.unsqueeze(1)
    kwargs["edge_apply_node"] = edge_apply_node

    edge_index = (
        (torch.arange(head.numel(), device=head.device) // 2)
        .unsqueeze(0)
        .expand_as(edge_apply_node)
    )
    kwargs["edge_index"] = edge_index

    return kwargs


def supporting_loss_func(
    graph_solved: "StructData",
    steps,
    n_nodes_full,
    n_edges_full,
    edge_apply_node,
    edge_index,
):
    # NOTE: this version assumes that an edge (even) and its reciprocal (odd) ara adjacent in the array
    # Init linear system matrix
    # TODO: Make sparse
    A_sys_full = torch.zeros([n_nodes_full * 3, n_edges_full], dtype=TORCH_FLOAT)
    # Populate linear system vector
    b_sys_full = graph_solved.load.T.reshape(-1)

    res_steps = torch.zeros([3 * n_nodes_full, len(steps)], dtype=TORCH_FLOAT)

    # Pupulate A_sys

    edge_tail_coords = graph_solved.coords[graph_solved.edge_index[0, :]]
    edge_head_coords = graph_solved.coords[graph_solved.edge_index[1, :]]

    edge_difference = edge_head_coords - edge_tail_coords
    edge_direction = edge_difference / torch.linalg.norm(
        edge_difference, dim=1, keepdim=True
    )

    A_sys_full.index_put_(
        (edge_apply_node.reshape(-1), edge_index.reshape(-1)),
        edge_direction.T.reshape(-1),
        accumulate=True,
    )

    # steps = [steps[-1]]
    for step in steps:
        # Find active edges
        active_edges_mask_directed = (graph_solved.assembly_sequence <= step).squeeze()
        active_edges_mask = active_edges_mask_directed[::2]
        # Find active nodes
        active_nodes_mask = torch.zeros(
            n_nodes_full, dtype=torch.bool, device=graph_solved.edge_index.device
        )
        active_nodes_mask[
            graph_solved.edge_index[:, active_edges_mask_directed].reshape(-1)
        ] = True
        active_nodes_mask[graph_solved.is_support.squeeze()] = False

        active_nodes_mask_full = active_nodes_mask.repeat(3)

        # Get subsystem
        A_sub = A_sys_full[active_nodes_mask_full][:, active_edges_mask]
        b_sub = b_sys_full[active_nodes_mask_full]

        # Solve subsystem
        beta = torch.linalg.solve(A_sub.T @ A_sub, A_sub.T @ b_sub)
        res_steps[active_nodes_mask_full, step] = A_sub @ beta - b_sub

    return torch.linalg.norm(res_steps)
