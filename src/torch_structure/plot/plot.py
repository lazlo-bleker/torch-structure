import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from torch_structure.message_passing import ResidualForce
from .plot_settings import PlotSettings

_FORCE_THRESHOLD = 1e-5


def plot_data(
    coords,
    edge_index,
    is_support=None,
    force=None,
    load=None,
    show_supports=True,
    show_nodes=True,
    show_node_labels=True,
    show_load=True,
    show_reactions=True,
    show_axes=True,
    legend=True,
    equal_axes=True,
    lw_constant=False,
    force_scale=1.0,
    lw_scale=1.0,
    settings=None,
    title=None,
    ax=None,
    path=None,
    show=False,
    show_edge_indices=False,
    show_deck=False,
    is_deck_node=None
):
    """
    Plot a structure in 3D.

    Args:
        coords (torch.Tensor): Tensor of shape (num_nodes, 3) with the 3D coordinates of each node.
        edge_index (torch.Tensor): Tensor of shape (2, num_edges) defining edge connections by node indices.
        is_support (torch.Tensor, optional): Boolean tensor of shape (num_nodes) indicating which nodes are supports. Needed for show_supports and show_reactions; silently skipped if omitted.
        force (torch.Tensor, optional): Tensor of shape (num_edges) with axial force values for each edge. Needed for show_reactions; silently skipped if omitted.
        load (torch.Tensor, optional): Tensor of shape (num_nodes, 3) specifying external loads applied to each node. Needed for show_load; silently skipped if omitted.
        show_supports (bool, optional): If True, support markers are drawn at nodes where is_support is True (requires `is_support`). Default is True.
        show_nodes (bool, optional): If True, a marker (white fill, black border) is drawn at every node. Default is True.
        show_node_labels (bool, optional): If True, each node is labeled with its index. Default is True.
        show_load (bool, optional): If True, non-zero external load vectors are plotted (requires `load`). Default is True.
        show_reactions (bool, optional): If True, non-zero support reaction vectors are plotted at support nodes (requires `force` and `is_support`). Default is True.
        show_axes (bool, optional): If True, axis lines and labels will be shown. Default is False.
        legend (bool, optional): If True, a legend will be displayed. Default is True.
        equal_axes (bool, optional): If True, sets equal scaling for all axes. Default is True.
        lw_constant (bool, optional): If True, uses uniform line widths for all edges. If False, widths are scaled by force magnitude. Default is False.
        force_scale (float, optional): Scale factor for visualizing force vectors (load and reaction). Default is 1.0.
        lw_scale (float, optional): Scale factor for line widths of edges. Default is 1.0.
        settings (PlotSettings, optional): Visual settings (colors, marker styles, sizes). Defaults to `PlotSettings()`.
        title (str, optional): Title for the plot. Default is None.
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): An existing 3D axis object to plot on. If None, a new one is created.
        path (str, optional): If provided, saves the figure as a PNG to this file path (without extension).
        show (bool, optional): If True, the plot will be displayed with plt.show(). Default is False.

    """
    if settings is None:
        settings = PlotSettings()

    force = force.view(-1) if force is not None else force
    is_support = is_support.view(-1) if is_support is not None else is_support

    coords_np = coords.detach().cpu().numpy()
    x = coords_np[:, 0]
    y = coords_np[:, 1]
    z = coords_np[:, 2]
    num_nodes = coords.size(dim=0)
    num_edges = edge_index.size(dim=1)

    if force is not None:
        force_np = force.detach().cpu().numpy()
        edge_color = np.where(force_np > 0, settings.edge_tension_color, settings.edge_compression_color)
        edge_label = np.where(force_np > 0, "Tension", "Compression")
        lw = np.sqrt(np.abs(force_np)) if not lw_constant else np.ones(num_edges)
        lw = lw * lw_scale
    else:
        edge_color = ["gray"] * num_edges
        lw = np.ones(num_edges) * lw_scale
        edge_label = ["Edge"] * num_edges

    # Prepare figure if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

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
            alpha=settings.edge_alpha,
            label=edge_label[i],
            zorder=settings.zorder_edge,
        )
        if show_edge_indices:
            mid_x = (x[src] + x[dst]) / 2
            mid_y = (y[src] + y[dst]) / 2
            mid_z = (z[src] + z[dst]) / 2
            ax.text(mid_x, mid_y, mid_z, str(i), color="black", fontsize=8, zorder=settings.zorder_label)

    # Plot node markers (uniform white fill, black border; support nodes get only the triangle marker below)
    if show_nodes:
        if is_support is not None:
            node_mask = ~is_support.detach().cpu().numpy().astype(bool)
        else:
            node_mask = np.ones(num_nodes, dtype=bool)
        if node_mask.any():
            ax.plot(
                x[node_mask], y[node_mask], z[node_mask],
                marker=settings.node_marker,
                markersize=settings.node_size,
                markerfacecolor=settings.node_color,
                markeredgecolor=settings.node_outline_color,
                markeredgewidth=settings.node_outline_width,
                linestyle="None",
                label="Node",
                zorder=settings.zorder_node,
            )

    # Plot node labels
    if show_node_labels:
        for node in range(num_nodes):
            ax.text(
                x[node], y[node], z[node] + settings.node_label_offset,
                str(node),
                color=settings.node_label_color,
                fontsize=settings.node_label_size,
                ha="center",
                va="center",
                zorder=settings.zorder_label,
            )

    # Plot supports
    if show_supports and is_support is not None:
        support_mask = is_support.detach().cpu().numpy().astype(bool)
        if support_mask.any():
            ax.plot(
                x[support_mask],
                y[support_mask],
                z[support_mask],
                marker=settings.support_marker,
                markersize=settings.support_marker_size,
                markerfacecolor=settings.support_marker_color,
                markeredgecolor=settings.support_marker_outline_color,
                markeredgewidth=settings.node_outline_width,
                linestyle="None",
                label="Support",
                zorder=settings.zorder_node,
            )

    # Points that must remain inside the view: node coords plus any vector
    # endpoint, since matplotlib's 3D quiver does not reliably participate
    # in axis autoscaling.
    extent_points = [coords_np]

    # Plot external load
    if show_load and load is not None:
        load_np = load.detach().cpu().numpy()
        for node in range(num_nodes):
            if np.linalg.norm(load_np[node]) <= _FORCE_THRESHOLD:
                continue
            d_x = force_scale * load_np[node, 0]
            d_y = force_scale * load_np[node, 1]
            d_z = force_scale * load_np[node, 2]
            tip = np.array([
                x[node] + d_x * settings.vector_length,
                y[node] + d_y * settings.vector_length,
                z[node] + d_z * settings.vector_length,
            ])
            extent_points.append(tip[None, :])
            ax.quiver(
                x[node],
                y[node],
                z[node],  # Base of the arrow
                d_x,
                d_y,
                d_z,  # Direction vector
                color=settings.vector_load_color,
                length=settings.vector_length,
                normalize=False,
                arrow_length_ratio=settings.vector_arrow_ratio,
                linewidth=settings.vector_line_width,
                alpha=settings.vector_alpha,
                label="Load",
                zorder=settings.zorder_vector,
            )

    # Plot support reactions
    if show_reactions and is_support is not None and force is not None:
        calculate_residual_force = ResidualForce()
        reaction_load = load if load is not None else torch.zeros_like(coords)
        reaction_forces = (
            calculate_residual_force(coords, force, edge_index, reaction_load)
            .detach()
            .cpu()
            .numpy()
        )
        for node in range(num_nodes):
            if not is_support[node]:
                continue
            if np.linalg.norm(reaction_forces[node]) <= _FORCE_THRESHOLD:
                continue
            d_x = force_scale * reaction_forces[node, 0]
            d_y = force_scale * reaction_forces[node, 1]
            d_z = force_scale * reaction_forces[node, 2]
            # Arrowhead lands on the support node, tail extends along the reaction
            # direction: quiver's tip = base + (u,v,w)*length, so with base on the
            # +d side of the node, (u,v,w) must point back toward the node (-d).
            base_x = x[node] + d_x * settings.vector_length
            base_y = y[node] + d_y * settings.vector_length
            base_z = z[node] + d_z * settings.vector_length
            extent_points.append(np.array([[base_x, base_y, base_z]]))
            ax.quiver(
                base_x,
                base_y,
                base_z,
                -d_x,
                -d_y,
                -d_z,
                color=settings.vector_reaction_color,
                length=settings.vector_length,
                normalize=False,
                arrow_length_ratio=settings.vector_arrow_ratio,
                linewidth=settings.vector_line_width,
                alpha=settings.vector_alpha,
                label="Support Reaction",
                zorder=settings.zorder_vector,
            )

    # Scale axes
    if equal_axes:
        all_points = np.vstack(extent_points)
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)

        x_range = maxs[0] - mins[0]
        x_middle = (mins[0] + maxs[0]) / 2
        y_range = maxs[1] - mins[1]
        y_middle = (mins[1] + maxs[1]) / 2
        z_range = maxs[2] - mins[2]
        z_middle = (mins[2] + maxs[2]) / 2

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

        ax.set_box_aspect([1, 1, 1])
        ax.set_proj_type("ortho")

    # Set axes visibility
    ax.set_axis_off()
    if show_axes:
        show_grid(ax, coords, minor_division=0.1, major_division=1.0, z=0.0)
        # Keep the cube zoomed to the structure; the grid is clipped to this view, not the reverse.
        if equal_axes:
            ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
            ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
            ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

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


def _round_up_to_nice_number(value, steps=(1, 2, 5, 10)):
    """Round a positive value up to the nearest "nice" number (1/2/5/10 x 10^n)."""
    if value <= 0:
        return float(steps[0])
    magnitude = 10 ** np.floor(np.log10(value))
    for step in steps:
        candidate = step * magnitude
        if candidate >= value - 1e-9:
            return float(candidate)
    return float(steps[-1] * magnitude)


def show_grid(
    ax,
    coords,
    minor_division=0.1,
    major_division=1.0,
    z=0.0,
    minor_color="#818181",
    major_color="#6E6E6E",
    minor_lw=0.3,
    major_lw=0.8,
    grid_alpha=0.1,
    x_axis_color="#964B4B",
    y_axis_color="#4B964B",
    axis_lw=1.5,
):
    """
    Draw a square reference grid on the XY plane of a 3D axis, sized to fit the model.

    The grid extent is derived from the model coordinates: the largest absolute x or y
    value is rounded up to the nearest "nice" number (1/2/5/10 x 10^n) and used as the
    (square) half-width of the grid. Minor lines are drawn every `minor_division` units,
    and lines that fall on a multiple of `major_division` are drawn thicker. The positive
    X and Y axes are highlighted from the origin to the edge of the grid.

    Args:
        ax (matplotlib.axes._subplots.Axes3DSubplot): 3D axis to draw the grid on.
        coords (torch.Tensor): Tensor of shape (num_nodes, 3) with the 3D coordinates of each node.
        minor_division (float, optional): Spacing between minor grid lines. Default is 0.1.
        major_division (float, optional): Spacing between major (thicker) grid lines. Default is 1.0.
        z (float, optional): Height at which to draw the grid. Default is 0.0.
        minor_color (str, optional): Color of minor grid lines.
        major_color (str, optional): Color of major grid lines.
        minor_lw (float, optional): Line width of minor grid lines.
        major_lw (float, optional): Line width of major grid lines.
        grid_alpha (float, optional): Opacity of the grid lines (not the X/Y axis lines). Default is 0.3.
        x_axis_color (str, optional): Color of the X axis line.
        y_axis_color (str, optional): Color of the Y axis line.
        axis_lw (float, optional): Line width of the X and Y axis lines.

    Returns:
        float: The computed grid extent (half-width of the square grid).
    """
    coords_np = coords.detach().cpu().numpy()
    max_val = np.max(np.abs(coords_np[:, :2]))
    extent = _round_up_to_nice_number(max_val)

    n_minor = int(round(extent / minor_division))
    lines = np.linspace(-extent, extent, 2 * n_minor + 1)
    major_step = int(round(major_division / minor_division))

    for i, v in enumerate(lines):
        is_major = i % major_step == 0
        color = major_color if is_major else minor_color
        lw = major_lw if is_major else minor_lw
        ax.plot([-extent, extent], [v, v], [z, z], color=color, lw=lw, alpha=grid_alpha, zorder=0)
        ax.plot([v, v], [-extent, extent], [z, z], color=color, lw=lw, alpha=grid_alpha, zorder=0)

    ax.plot([0, extent], [0, 0], [z, z], color=x_axis_color, lw=axis_lw, zorder=1)
    ax.plot([0, 0], [0, extent], [z, z], color=y_axis_color, lw=axis_lw, zorder=1)

    return extent


def show_grid_xz(
    ax,
    coords,
    minor_division=0.1,
    major_division=1.0,
    minor_color="#4D4D4D",
    major_color="#4D4D4D",
    minor_lw=0.3,
    major_lw=0.8,
    grid_alpha=0.3,
    x_axis_color="#964B4B",
    z_axis_color="#4B4B96",
    axis_lw=1.5,
):
    """
    Draw a square reference grid on a 2D X-Z elevation axis, sized to fit the model.

    The grid extent is derived from the model coordinates: the largest absolute x or z
    value is rounded up to the nearest "nice" number (1/2/5/10 x 10^n) and used as the
    (square) half-width of the grid. Minor lines are drawn every `minor_division` units,
    and lines that fall on a multiple of `major_division` are drawn thicker. The positive
    X and Z axes are highlighted from the origin to the edge of the grid.

    Args:
        ax (matplotlib.axes.Axes): 2D axis to draw the grid on.
        coords (torch.Tensor): Tensor of shape (num_nodes, 3) with the 3D coordinates of each node.
        minor_division (float, optional): Spacing between minor grid lines. Default is 0.1.
        major_division (float, optional): Spacing between major (thicker) grid lines. Default is 1.0.
        minor_color (str, optional): Color of minor grid lines.
        major_color (str, optional): Color of major grid lines.
        minor_lw (float, optional): Line width of minor grid lines.
        major_lw (float, optional): Line width of major grid lines.
        grid_alpha (float, optional): Opacity of the grid lines (not the X/Z axis lines). Default is 0.3.
        x_axis_color (str, optional): Color of the X axis line.
        z_axis_color (str, optional): Color of the Z axis line.
        axis_lw (float, optional): Line width of the X and Z axis lines.

    Returns:
        float: The computed grid extent (half-width of the square grid).
    """
    coords_np = coords.detach().cpu().numpy()
    max_val = np.max(np.abs(coords_np[:, [0, 2]]))
    extent = _round_up_to_nice_number(max_val)

    n_minor = int(round(extent / minor_division))
    lines = np.linspace(-extent, extent, 2 * n_minor + 1)
    major_step = int(round(major_division / minor_division))

    for i, v in enumerate(lines):
        is_major = i % major_step == 0
        color = major_color if is_major else minor_color
        lw = major_lw if is_major else minor_lw
        ax.plot([-extent, extent], [v, v], color=color, lw=lw, alpha=grid_alpha, zorder=0)
        ax.plot([v, v], [-extent, extent], color=color, lw=lw, alpha=grid_alpha, zorder=0)

    ax.plot([0, extent], [0, 0], color=x_axis_color, lw=axis_lw, zorder=1)
    ax.plot([0, 0], [0, extent], color=z_axis_color, lw=axis_lw, zorder=1)

    return extent


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
    ax.set_axis_off()
    if show_axes:
        show_grid_xz(ax, coords, minor_division=0.1, major_division=1.0)

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
