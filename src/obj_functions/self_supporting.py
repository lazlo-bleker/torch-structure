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
    return kwargs


def _eval_projection_systems(
    graph_solved: "StructData",
    steps,
    n_nodes_full,
    n_edges_full,
):
    # NOTE: this version assumes that an edge (even) and its reciprocal (odd) ara adjacent in the array
    # Init linear system matrix
    # TODO: Make sparse
    A_sys_full = torch.zeros([n_nodes_full * 3, n_edges_full], dtype=TORCH_FLOAT)
    # Populate linear system vector
    b_sys_full = graph_solved.load.T.reshape(-1)

    beta_steps = torch.zeros([n_edges_full, len(steps)], dtype=TORCH_FLOAT)
    res_steps = torch.zeros([3 * n_nodes_full, len(steps)], dtype=TORCH_FLOAT)

    # Pupulate A_sys
    for edge_number_directed, edge_indices in enumerate(graph_solved.edge_index.T):
        # NOTE: Changes to node coords will change directions (carry gradients)
        edge_direction = _get_edge_direction(graph_solved, edge_number_directed)
        edge_number = edge_number_directed // 2
        head_node_indices = [
            edge_indices[1],
            edge_indices[1] + n_nodes_full,
            edge_indices[1] + 2 * n_nodes_full,
        ]
        A_sys_full[head_node_indices, edge_number] += edge_direction

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
        beta_steps[active_edges_mask, step] = beta
        res_steps[active_nodes_mask_full, step] = A_sub @ beta - b_sub

    return beta_steps, res_steps


def supporting_loss_plot(
    graph_solved: "StructData",
    steps,
    n_nodes_full,
    n_edges_full,
    suffix=str,
    to_gif=True,
):
    # Init dict
    beta_steps, res_steps = _eval_projection_systems(
        graph_solved, steps, n_nodes_full, n_edges_full
    )
    _plot_supporting_loss_state(
        graph_solved,
        beta_steps,
        res_steps,
        suffix=suffix,
        to_gif=to_gif,
    )

    return beta_steps, res_steps


def supporting_loss_func(
    graph_solved: "StructData",
    steps,
    n_nodes_full,
    n_edges_full,
):
    _, res_steps = _eval_projection_systems(
        graph_solved, steps, n_nodes_full, n_edges_full
    )
    return torch.linalg.norm(res_steps)


def _plot_supporting_loss_state(
    graph_solved: "StructData",
    beta_steps,
    res_steps,
    suffix="",
    to_gif=True,
):

    work_dir = f"./img/{suffix}"
    os.makedirs(work_dir, exist_ok=True)

    for step in range(beta_steps.shape[1]):
        step_force_dual = torch.zeros(graph_solved.num_edges)
        step_force_dual[::2] = beta_steps[:, step]
        step_force_dual[1::2] = beta_steps[:, step]
        graph_solved.plot(
            load=res_steps[:, step].reshape([3, -1]).T,
            force=step_force_dual.unsqueeze(1),
            show_load=True,
            force_scale=5e1,
            path=f"{work_dir}/state_{step:03d}",
            title=f"Aux. Force Loss: {torch.linalg.norm(res_steps[:, step]):.2e}",
        )
        plt.close()

    if to_gif:
        _save_as_gif(suffix)


def _save_as_gif(suffix):
    # Folder containing your images
    folder_path = f"./img/{suffix}"
    output_gif = f"./img/{suffix}.gif"

    # Collect all image files (sorted)
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            images.append(imageio.imread(image_path))

    # Save as GIF
    imageio.mimsave(
        output_gif, images, duration=0.5
    )  # duration = time per frame in seconds
    shutil.rmtree(folder_path)


def _get_edge_direction(graph_solved: "StructData", edge_index):
    if edge_index == -1:
        return torch.zeros(3, dtype=TORCH_FLOAT)
    tail, head = graph_solved.coords[graph_solved.edge_index[:, edge_index]]
    direction = head - tail
    direction = direction / torch.linalg.norm(direction, keepdim=True)
    return direction
