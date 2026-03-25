from torch_structure.data import StructData
import torch
from config import TORCH_FLOAT
import matplotlib.pyplot as plt

def cache_supporting_loss(graph_solved: "StructData"):
    kwargs = {}
    n_steps = int(max(graph_solved.assembly_sequence))+1
    n_edges_full = graph_solved.num_edges // 2 # Account for a directed graph
    n_nodes_full = graph_solved.num_nodes
    kwargs["n_steps"] = n_steps
    kwargs["log_internal_forces"] = torch.zeros((n_steps, n_edges_full), dtype=TORCH_FLOAT)
    kwargs["log_auxiliary_forces"] = torch.zeros((n_steps, n_nodes_full, 3), dtype=TORCH_FLOAT) 

    return kwargs

def plot_supporting_loss(graph_solved: "StructData", step_losses, n_steps=None, log_internal_forces=None, log_auxiliary_forces=None):
    graph_solved.plot(path=f"./img/final.png")
    for step in range(n_steps):
        step_force_dual = torch.zeros(graph_solved.num_edges)
        step_force_dual[::2] = log_internal_forces[step]
        step_force_dual[1::2] = log_internal_forces[step]
        graph_solved.plot(
             load=log_auxiliary_forces[step],
             force=step_force_dual.unsqueeze(1),
             show_load=True,
            #  force_scale = 8e0,
             force_scale = 1e2,
             path=f"./img/state_{step:03d}.png",
             title=f"Aux. Force Loss: {step_losses[step]:.2e}"
        )
        plt.close()

def supporting_loss(graph_solved: "StructData", n_steps=None, log_internal_forces=None, log_auxiliary_forces=None):
    # NOTE: this version assumes that an edge (even) and its reciprocal (odd) ara adjacent in the array
    step_losses = torch.zeros(n_steps, dtype=TORCH_FLOAT)
    steps = torch.arange(n_steps)
    # System mapping edge forces to node forces (flattened)
    # Initialize array
    n_edges_full = graph_solved.num_edges // 2 # Account for a directed graph
    n_nodes_full = graph_solved.num_nodes
    A_sys_full = torch.zeros([n_nodes_full * 3, n_edges_full], dtype=TORCH_FLOAT)
    # Pupulate b_sys
    b_sys_full = graph_solved.load.T.reshape(-1)
    # Pupulate A_sys
    for edge_number_directed, edge_indices in enumerate(graph_solved.edge_index.T):
        edge_direction =  _get_edge_direction(graph_solved, edge_number_directed)
        edge_number = edge_number_directed // 2
        head_node_indices = [edge_indices[1], edge_indices[1] + n_nodes_full, edge_indices[1] + 2 * n_nodes_full]
        A_sys_full[head_node_indices, edge_number] += edge_direction
 
    # steps = [steps[-1]]
    for step in steps:
        # Find active edges
        active_edges_mask_directed = (graph_solved.assembly_sequence <= step).squeeze()
        active_edges_mask = active_edges_mask_directed[::2]
        # Find active nodes
        active_nodes_mask = torch.zeros(n_nodes_full, dtype=torch.bool, device=graph_solved.edge_index.device)
        active_nodes_mask[graph_solved.edge_index[:, active_edges_mask_directed].reshape(-1)] = True
        active_nodes_mask[graph_solved.is_support.squeeze()]= False

        active_nodes_mask_full = active_nodes_mask.repeat(3)

        # Get subsystem
        A_sub = A_sys_full[active_nodes_mask_full][:, active_edges_mask]
        b_sub = b_sys_full[active_nodes_mask_full]

        # Solve subsystem
        beta = torch.linalg.solve(A_sub.T @ A_sub, A_sub.T @ b_sub)
        res = A_sub @ beta - b_sub

        # Add loss contribution
        step_losses[step] = torch.linalg.norm(res)

        # Add data to log
        log_internal_forces[step][active_edges_mask] = beta
        log_auxiliary_forces[step][active_nodes_mask] = - res.reshape(3,-1).T

    print(step_losses)
    return torch.sum(step_losses), step_losses

def _get_edge_direction(graph_solved: "StructData", edge_index):
        if edge_index == -1:
             return torch.zeros(3, dtype=TORCH_FLOAT)
        tail, head = graph_solved.coords[graph_solved.edge_index[:,edge_index]]
        direction = head - tail
        direction /= torch.linalg.norm(direction)
        return direction