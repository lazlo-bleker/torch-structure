import matplotlib.pyplot as plt
import numpy as np

from torch_structure.message_passing import ResidualForce


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

    colors = {"red": "#E40714", "blue": "#0578BF", "green": "#007F00"}  # Todo: move to config

    coords_np = coords.detach().cpu().numpy()
    x = coords_np[:, 0]
    y = coords_np[:, 1]
    z = coords_np[:, 2]
    num_nodes = coords.size(dim=0)
    num_edges = edge_index.size(dim=1)

    if force is not None:
        force_np = force.detach().cpu().numpy()
        edge_color = np.where(force_np > 0, colors["red"], colors["blue"])
        edge_label = np.where(force_np > 0, "Tension (Predicted Equilibrium Geometry)", "Compression (Predicted Equilibrium Geometry)")
        lw = np.sqrt(np.abs(force_np)) if not lw_constant else np.ones(num_edges)
        lw = lw * lw_scale
    else:
        edge_color = ["gray"] * num_edges
        lw = np.ones(num_edges) * lw_scale
        edge_label = ["Target Geometry"] * num_edges

    # Prepare figure if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

    # Plot supports
    if show_supports:
        if is_support is None:
            raise ValueError("For showing supports is_support is a required input.")
        for node in range(num_nodes):
            if is_support[node]:
                if node in edge_index[0].cpu().numpy():
                    triangle_size = support_marker_offset
                    ax.plot(
                        [x[node]], [y[node]], [z[node] - triangle_size],
                        marker="^",
                        markersize=support_marker_size,
                        color="black",
                        linestyle="None"
                    )

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

    # Plot external load
    if show_load:
        if load is None:
            raise ValueError("For showing external forces load is a required input.")
        load_np = load.detach().cpu().numpy()
        for node in range(num_nodes):
            d_x = force_scale * load_np[node, 0]
            d_y = force_scale * load_np[node, 1]
            d_z = force_scale * load_np[node, 2]
            ax.plot(
                [x[node], x[node] + d_x],
                [y[node], y[node] + d_y],
                [z[node], z[node] + d_z],
                color=colors["green"],
                label="External Load",
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
            d_y = force_scale * residual_forces[node, 1]
            d_z = force_scale * residual_forces[node, 2]
            ax.plot(
                [x[node], x[node] + d_x],
                [y[node], y[node] + d_y],
                [z[node], z[node] + d_z],
                color="purple",
                label="Residual Force",
            )

    # Set axes visibility
    if not show_axes:
        ax.set_axis_off()

    # Scale axes
    if equal_axes:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        ax.set_box_aspect([1,1,1])
        ax.set_proj_type('ortho')

    # Set legend
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), frameon=False)

    # Set title
    if title is not None:
        ax.set_title(title)

    # Save file if path is provided
    if path is not None:
        plt.savefig((f"{path}.png"), bbox_inches="tight")
    
    # Show plot
    if show:
        plt.show()
