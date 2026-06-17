import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
from torch_structure.data import StructData
from torch_geometric.utils import to_networkx
from post.export_vtp import export_graph_to_vtp
from post.export_img import save_as_gif, save_as_video


def export_aggragate_vtp(post_data: StructData, export_dir, base_name, cache_dict_vtp):
    filename = base_name + "_aggregate"
    graph = to_networkx(
        post_data,
        node_attrs=["coords", "aux_force_total", "laplacian"],
        edge_attrs=[],
        to_undirected=True,
    )
    export_graph_to_vtp(graph, export_dir / "paraview" / f"{filename}.vtp")


def export_aggragate_img(post_data: StructData, export_dir, base_name, cache_dict_img):
    # Dummy, does not actually plot, since it does not have all the data
    # Instead, it collects all the data and a breakpoint is used to plot
    profile = torch.sum(
        torch.linalg.norm(post_data.aux_force_steps, dim=2), dim=0
    ).tolist()
    if "profiles" in cache_dict_img.keys():
        assert type(cache_dict_img["profiles"]) == list
        cache_dict_img["profiles"].append(profile)
    else:
        cache_dict_img["profiles"] = [profile]


def export_assembly_states_vtp(
    post_data: StructData, export_dir, base_name, cache_dict_vtp
):
    n_steps = post_data.aux_force_steps.shape[1]
    for step in range(n_steps):
        filename = base_name + f"state_{step:05d}.vtp"
        (export_dir / "paraview").mkdir(parents=True, exist_ok=True)
        post_data.aux_force = post_data.aux_force_steps[:, step]
        post_data.internal_force = post_data.internal_force_steps[:, step]
        graph = to_networkx(
            post_data,
            node_attrs=["coords", "aux_force"],
            edge_attrs=["internal_force"],
            to_undirected=True,
        )
        export_graph_to_vtp(
            graph,
            export_dir / "paraview" / filename,
        )


def export_assembly_states_img(
    post_data: StructData, export_dir, base_name="", to_gif=True
):
    aux_forces_mag = torch.linalg.norm(post_data.aux_force_steps, dim=2)
    residuals = torch.sum(aux_forces_mag, dim=0)
    steps = torch.arange(post_data.aux_force_steps.shape[1])
    n_bricks = torch.zeros_like(steps)
    for i in range(len(steps)):
        mask = post_data.assembly_sequence <= i
        n_bricks[i] = torch.sum(mask)
    plt.rcParams["font.size"] = 16

    plt.figure(figsize=(8, 4))
    plt.plot(n_bricks, residuals, marker="o", color="darkorange")
    plt.xlabel("Active Element Count")
    plt.ylabel("Auxiliary Forces Norm")
    # plt.yscale('log')  # Residuals are often best on a log scale
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title("Support Auxiliary Forces")
    plt.tight_layout()

    # Save (or show) the figure
    filename = f"loss_profile.png"
    # Implement image rendering
    filepath = export_dir / filename

    plt.savefig(filepath, dpi=150)

    n_steps = post_data.aux_force_steps.shape[1]
    for step in range(n_steps):
        filename = base_name + f"state_{step:05d}.png"
        (export_dir / "img").mkdir(parents=True, exist_ok=True)
        post_data.aux_force = post_data.aux_force_steps[:, step]
        post_data.internal_force = post_data.internal_force_steps[:, step]
        # Implement image rendering
        filepath = export_dir / "img" / filename
        post_data.plot(
            path=filepath,
            force=post_data.internal_force,
            load=post_data.aux_force,
            show_load=True,
            force_scale=1e0,
        )
        plt.close()

    if to_gif:
        # image-to-gif conversion
        save_as_gif(export_dir / "img", base_name + f"animation.gif")
        pass
