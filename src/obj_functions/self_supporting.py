from torch_structure.data import StructData
import torch
from config import TORCH_FLOAT, DEVICE


def supporting_loss_cache(graph_solved: "StructData"):
    kwargs = {}
    # Validate that the assembly sequence starts at 0
    graph_solved.assembly_sequence -= int(min(graph_solved.assembly_sequence))
    kwargs["n_nodes_full"] = graph_solved.num_nodes
    n_edges_directed = graph_solved.num_edges // 2
    kwargs["n_edges_full"] = n_edges_directed  # Account for a directed graph
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
    
    edge_index = torch.zeros_like(edge_apply_node)
    
    _edge_index =  torch.arange(n_edges_directed, dtype=torch.long, device=DEVICE).unsqueeze(0)
    edge_index[:, graph_solved.directed_mask.squeeze(1)] = _edge_index
    edge_index[:,~graph_solved.directed_mask.squeeze(1)] = _edge_index
    kwargs["edge_index"] = edge_index

    return kwargs


def _eval_auxiliary_forces(
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
    A_sys_full = torch.zeros(
        [n_nodes_full * 3, n_edges_full], dtype=TORCH_FLOAT, device=DEVICE
    )
    # Populate linear system vector
    b_sys_full = graph_solved.load.T.reshape(-1)

    aux_forces_steps = torch.zeros(
        [n_nodes_full, len(steps), 3], dtype=TORCH_FLOAT, device=DEVICE
    )
    internal_force_steps = torch.zeros(
        [n_edges_full, len(steps)], dtype=TORCH_FLOAT, device=DEVICE
    )

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

    for step in steps:
        # Find active edges
        supported_nodes = graph_solved.is_support.squeeze()
        both_supported = torch.all(supported_nodes[graph_solved.edge_index], dim=0)
        active_sequence = (graph_solved.assembly_sequence <= step).squeeze()
        active_edges_mask_directed = active_sequence & ~both_supported
        active_edges_mask = active_edges_mask_directed[graph_solved.directed_mask.squeeze(1)]
        # Find active nodes
        active_nodes_mask = torch.zeros(
            n_nodes_full, dtype=torch.bool, device=graph_solved.edge_index.device
        )
        active_nodes_mask[
            graph_solved.edge_index[:, active_edges_mask_directed].reshape(-1)
        ] = True
        active_nodes_mask[supported_nodes] = False

        active_nodes_mask_full = active_nodes_mask.repeat(3)

        # Get subsystem
        A_sub = A_sys_full[active_nodes_mask_full][:, active_edges_mask]
        b_sub = b_sys_full[active_nodes_mask_full]

        # Solve subsystem
        beta = torch.linalg.solve(A_sub.T @ A_sub, A_sub.T @ b_sub)
        aux_forces_flat = A_sub @ beta - b_sub
        internal_force_steps[active_edges_mask, step] = beta
        aux_forces_steps[active_nodes_mask, step] = aux_forces_flat.reshape((3, -1)).T

    return aux_forces_steps, internal_force_steps


def supporting_loss_func(*args, **kwargs):
    aux_forces_steps, _ = _eval_auxiliary_forces(*args, **kwargs)
    return torch.linalg.norm(aux_forces_steps)


def graph_post_process(graph):
    kwargs = supporting_loss_cache(graph_solved=graph)
    aux_forces_steps, internal_force_steps = _eval_auxiliary_forces(graph, **kwargs)
    internal_force_steps = torch.repeat_interleave(
        internal_force_steps, repeats=2, dim=0
    )
    aux_force_total = torch.sum(torch.linalg.norm(aux_forces_steps, dim=2), dim=1)

    setattr(graph, "aux_force_steps", aux_forces_steps)
    setattr(graph, "aux_force_total", aux_force_total)
    setattr(graph, "internal_force_steps", internal_force_steps)

    return graph
