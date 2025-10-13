import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from torch_structure.message_passing import ResidualForce

elev_default, azim_default, roll_default = (20.0, 90.0, 0.0)

def plot_data(
    coords,
    edge_index,
    is_support=None,
    force=None,
    load=None,
    show_supports=False,
    show_load=False,
    show_residual_forces=False,
    show_axes=False,
    legend=True,
    equal_axes=True,
    lw_constant=False,
    force_scale=1.0,
    lw_scale=1.0,
    support_marker_size=6,
    support_marker_offset=0.02,
    title=None,
    ax=None,
    path=None,
    show=False,
    out=True,
    elev_default=20,
    azim_default=-60,
    roll_default=0,
):
    """
    Same signature as your plot_data, but much faster by:
    - drawing edges with a single Line3DCollection
    - using vectorized numpy ops
    - building the legend from proxy handles
    - calling quiver once with array inputs
    """

    # --- tensor -> numpy once
    coords_np = coords.detach().cpu().numpy()
    edges_np = edge_index.detach().cpu().numpy().T  # (E, 2)
    x, y, z = coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]
    num_nodes = coords_np.shape[0]
    num_edges = edges_np.shape[0]

    colors = {
        "red": "#E40714",
        "blue": "#0578BF",
        "green": "#007F00",
    }

    # --- force-dependent styling (vectorized)
    if force is not None:
        force_np = force.detach().cpu().ravel().numpy()
        edge_colors = np.where(force_np > 0, colors["red"], colors["blue"])
        lw = (np.sqrt(np.abs(force_np)) if not lw_constant else np.ones(num_edges)) * lw_scale
        has_force = True
    else:
        edge_colors = np.array(["gray"] * num_edges)
        lw = np.full(num_edges, lw_scale)
        has_force = False

    # --- prepare axes once
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev_default, azim=azim_default, roll=roll_default)
        created_fig = True

    # --- build edge segments for a single draw call: (E, 2, 3)
    seg_start = coords_np[edges_np[:, 0]]  # (E, 3)
    seg_end   = coords_np[edges_np[:, 1]]  # (E, 3)
    segments = np.stack([seg_start, seg_end], axis=1)  # (E, 2, 3)

    lc = Line3DCollection(segments, colors=edge_colors.tolist(), linewidths=lw, antialiased=False)
    ax.add_collection3d(lc)

    # --- supports (single scatter instead of a loop)
    if show_supports:
        if is_support is None:
            raise ValueError("For showing supports is_support is a required input.")
        is_support_np = is_support.detach().cpu().ravel().numpy().astype(bool)
        if np.any(is_support_np):
            xs = x[is_support_np]
            ys = y[is_support_np]
            zs = z[is_support_np] - support_marker_offset
            # scatter uses points^2 for size:
            ax.scatter(xs, ys, zs, marker="^", s=(support_marker_size**2), c="black", depthshade=False)

    if show_deck:
        if is_deck_node is None:
            raise ValueError("For showing deck is_deck_node is a required input.")
        deck_coords = coords[is_deck_node.view(-1)].detach().cpu().numpy()
        # --- 1) sort by x
        deck_coords = deck_coords[np.argsort(deck_coords[:, 0])]

        # --- 2) enforce pairwise y ordering within each consecutive pair
        if len(deck_coords) % 2 != 0:
            raise ValueError("Expected an even number of deck coordinates (strict pairs).")
        for i in range(0, len(deck_coords), 2):
            if deck_coords[i, 1] > deck_coords[i+1, 1]:
                deck_coords[[i, i+1]] = deck_coords[[i+1, i]]

        # --- 3) group into pairs: shape -> (n_pairs, 2, 3)
        pairs = deck_coords.reshape(-1, 2, 3)  # [ [low_y, high_y] per x ]

        # --- 4) build quads between adjacent x-pairs
        quads = []
        for k in range(pairs.shape[0] - 1):
            left_low,  left_high  = pairs[k, 0], pairs[k, 1]
            right_low, right_high = pairs[k+1, 0], pairs[k+1, 1]

            # Counter-clockwise ordering (as seen from +x toward origin) to make normals consistent
            quad = [left_low, left_high, right_high, right_low]
            quads.append(quad)

        # --- 5) plot as a single Poly3DCollection
        coll = Poly3DCollection(quads, facecolors='grey', edgecolors='k', linewidths=0.5, alpha=0.4)
        ax.add_collection3d(coll)

    # Plot edges
    for i, (src, dst) in enumerate(edge_index.t().cpu().numpy()):
        ax.plot(
            [x[src], x[dst]],
            [y[src], y[dst]],
            [z[src], z[dst]],
            color=edge_color[i],
            lw=lw[i],
            label=edge_label[i],
        )
        if show_edge_indices:
            mid_x = (x[src] + x[dst]) / 2
            mid_y = (y[src] + y[dst]) / 2
            mid_z = (z[src] + z[dst]) / 2
            ax.text(mid_x, mid_y, mid_z, str(i), color="black", fontsize=8)

    # Plot external load
    if show_load:
        if load is None:
            raise ValueError("For showing external forces load is a required input.")
        load_np = load.detach().cpu().numpy()
        dx = force_scale * load_np[:, 0]
        dy = force_scale * load_np[:, 1]
        dz = force_scale * load_np[:, 2]
        ax.quiver(
            x, y, z,
            dx, dy, dz,
            color=colors["green"],
            length=1.5,
            normalize=False,
            arrow_length_ratio=0.5,
        )

    # --- residual forces (vectorized + one plot)
    if show_residual_forces:
        if load is None:
            raise ValueError("For showing residual forces load is a required input.")
        calculate_residual_force = ResidualForce()
        residual_np = calculate_residual_force(coords, force, edge_index, load).detach().cpu().numpy()
        ax.quiver(
            x, y, z,
            force_scale * residual_np[:, 0],
            force_scale * residual_np[:, 1],
            force_scale * residual_np[:, 2],
            color="purple",
            length=1.5,
            normalize=False,
            arrow_length_ratio=0.5,
        )

    # --- axes style
    if not show_axes:
        ax.set_axis_off()

    # --- set limits once from data (faster than hitting autoscale repeatedly)
    if equal_axes:
        mins = coords_np.min(axis=0)
        maxs = coords_np.max(axis=0)
        centers = 0.5 * (mins + maxs)
        ranges = (maxs - mins)
        radius = 0.5 * ranges.max()
        if radius == 0:  # degenerate case
            radius = 1.0
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)
        ax.set_box_aspect((1, 1, 1), zoom=1.5)
        ax.set_proj_type("persp")

    # --- legend using proxies (no per-edge labels)
    if legend:
        handles = []
        if has_force:
            handles.append(Line2D([0], [0], color=colors["red"],  lw=2, label="Tension (Predicted Equilibrium Geometry)"))
            handles.append(Line2D([0], [0], color=colors["blue"], lw=2, label="Compression (Predicted Equilibrium Geometry)"))
        else:
            handles.append(Line2D([0], [0], color="gray", lw=2, label="Target Geometry"))
        if show_load:
            handles.append(Line2D([0], [0], color=colors["green"], lw=2, label="External Load"))
        if show_residual_forces:
            handles.append(Line2D([0], [0], color="purple", lw=2, label="Residual Force"))
        ax.legend(handles=handles, frameon=False, loc="best")

    if title is not None:
        ax.set_title(title)

    if path is not None:
        plt.savefig(f"{path}.png", bbox_inches="tight")

    if show:
        plt.show()

    return ax if out else None

def plot_data_xz(
    coords,
    edge_index,
    is_support=None,
    force=None,
    load=None,
    show_supports=False,
    show_load=False,
    show_residual_forces=False,
    show_axes=False,
    legend=True,
    equal_axes=True,
    lw_constant=False,
    force_scale=0.6,
    lw_scale=1.0,
    support_marker_size=3,
    support_marker_offset=0.12,
    title=None,
    ax=None,
    path=None,
    show=False,
    support_marker_color="black",
    highlight_nodes=None,
):
    """
    Plot a structure in 3D.

    Args:
        coords (torch.Tensor): Tensor of shape (num_nodes, 3) with the 3D coordinates of each node.
        edge_index (torch.Tensor): Tensor of shape (2, num_edges) defining edge connections by node indices.
        is_support (torch.Tensor, optional): Boolean tensor of shape (num_nodes) indicating which nodes are supports. Required if show_supports=True.
        force (torch.Tensor, optional): Tensor of shape (num_edges) with axial force values for each edge.
        load (torch.Tensor, optional): Tensor of shape (num_nodes, 3) specifying external loads applied to each node. Required if show_load or show_residual_forces is True.
        show_supports (bool, optional): If True, support markers will be shown at nodes where is_support is True. Default is False.
        show_load (bool, optional): If True, external load vectors will be plotted. Requires `load`. Default is False.
        show_residual_forces (bool, optional): If True, plots residual force vectors. Requires `load`. Default is False.
        show_axes (bool, optional): If True, axis lines and labels will be shown. Default is False.
        legend (bool, optional): If True, a legend will be displayed. Default is True.
        equal_axes (bool, optional): If True, sets equal scaling for all axes. Default is True.
        lw_constant (bool, optional): If True, uses uniform line widths for all edges. If False, widths are scaled by force magnitude. Default is False.
        force_scale (float, optional): Scale factor for visualizing force vectors (load and residual). Default is 1.0.
        lw_scale (float, optional): Scale factor for line widths of edges. Default is 1.0.
        support_marker_size (int, optional): Marker size for support symbols. Default is 6.
        support_marker_offset (float, optional): Vertical offset for support markers (to avoid overlap with structure). Default is 0.02.
        title (str, optional): Title for the plot. Default is None.
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): An existing 3D axis object to plot on. If None, a new one is created.
        path (str, optional): If provided, saves the figure as a PNG to this file path (without extension).
        show (bool, optional): If True, the plot will be displayed with plt.show(). Default is False.

    """
    force = force.view(-1) if force is not None else force
    is_support = is_support.view(-1) if is_support is not None else is_support

    colors = {
        "red": "#E40714",
        "blue": "#0578BF",
        "green": "#007F00",
    }  # Todo: move to config

    coords_np = coords.detach().cpu().numpy()
    x = coords_np[:, 0]
    z = coords_np[:, 2]
    num_nodes = coords.size(dim=0)
    num_edges = edge_index.size(dim=1)

    if force is not None:
        force_np = force.detach().cpu().numpy()
        edge_color = np.where(force_np > 0, colors["red"], colors["blue"])
        edge_label = np.where(
            force_np > 0,
            "Tension",
            "Compression",
        )
        lw = np.sqrt(np.abs(force_np)) if not lw_constant else np.ones(num_edges)
        lw = lw * lw_scale
        linestyle = ["-"] * num_edges
    else:
        edge_color = ["gray"] * num_edges
        lw = np.ones(num_edges) * lw_scale
        edge_label = ["Converged Geometry"] * num_edges
        linestyle = [":"] * num_edges

    # Prepare figure if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

    # Plot edges
    plotted = []
    for i, (src, dst) in enumerate(edge_index.t().cpu().numpy()):
        if (dst, src) in plotted:
            continue
        ax.plot(
            [x[src], x[dst]],
            [z[src], z[dst]],
            color=edge_color[i],
            lw=lw[i],
            label=edge_label[i],
            linestyle=linestyle[i],
        )
        plotted.append((src, dst))

    # Plot external load
    if show_load:
        if load is None:
            raise ValueError("For showing external forces load is a required input.")
        load_np = load.detach().cpu().numpy()
        for node in range(num_nodes):
            d_x = force_scale * load_np[node, 0]
            d_z = force_scale * load_np[node, 2]
            if max(abs(d_x), abs(d_z)) > 1e-5:
                ax.plot(
                    [x[node], x[node] + d_x * 0.01],
                    [z[node], z[node] + d_z * 0.01],
                    color=colors["green"],
                    label="External Load",
                )
                ax.arrow(
                    x[node],
                    z[node],  # start point (x, y)
                    d_x,
                    d_z,  # delta (dx, dz)
                    color=colors["green"],
                    head_width=0.1,  # adjust as needed
                    head_length=0.1,  # adjust as needed
                    length_includes_head=True,
                )

    # Plot supports
    if show_supports:
        if is_support is None:
            raise ValueError("For showing supports is_support is a required input.")
        for node in range(num_nodes):
            if is_support[node]:
                if node in edge_index[1].cpu().numpy():  # TEMP FIX
                    triangle_size = support_marker_offset
                    ax.plot(
                        [x[node]],
                        [z[node] - triangle_size],
                        marker="^",
                        markersize=support_marker_size,
                        color=support_marker_color,
                        linestyle="None",
                        label="Support",
                    )

    # Plot residual forces
    if show_residual_forces:
        if load is None:
            raise ValueError("For showing residual forces load is a required input.")
        calculate_residual_force = ResidualForce()
        residual_forces = (
            calculate_residual_force(coords, force, edge_index, load)
            .detach()
            .cpu()
            .numpy()
        )
        for node in range(num_nodes):
            d_x = force_scale * residual_forces[node, 0]
            d_z = force_scale * residual_forces[node, 2]
            ax.plot(
                [x[node], x[node] + d_x],
                [z[node], z[node] + d_z],
                color="purple",
                label="Residual Force",
            )

    if highlight_nodes is not None:
        for node in highlight_nodes:
            ax.plot(
                [x[node]],
                [z[node]],
                marker="o",
                markersize=4,
                color="#FFEC99",
                linestyle="None",
                label="Considered in Step",
            )

    # Set axes visibility
    if not show_axes:
        ax.set_axis_off()

    # Scale axes
    if equal_axes:
        ax.set_aspect("equal", adjustable="box")

    # Set legend
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        order = [3, 2, 4, 0, 1, 5]
        unique_labels = {
            k: unique_labels[k] for k in np.array(list(unique_labels.keys()))[order]
        }
        ax.legend(
            unique_labels.values(),
            unique_labels.keys(),
            frameon=False,
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, -2.8),
        )

    # Set title
    if title is not None:
        ax.set_title(title, pad=20, fontsize=16)

    # Save file if path is provided
    if path is not None:
        plt.savefig((f"{path}.png"), bbox_inches="tight")

    # Show plot
    if show:
        plt.show()
