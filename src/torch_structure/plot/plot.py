import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from torch_structure.message_passing import ResidualForce
from torch_structure.plot.config import PLOT_CONFIG
from torch_structure.plot.primitives import (
    draw_arrow,
    draw_edge,
    draw_face,
    draw_node,
    draw_text,
)

# ResidualForce has no parameters and no expensive init (just aggr="add"), so
# there's nothing to gain from constructing it lazily — one shared instance.
calculate_residual_force = ResidualForce()


def _residual_forces(coords, force, edge_index, load):
    """
    Compute the residual (out-of-equilibrium) force at every node.

    Args:
        coords (torch.Tensor): Tensor of shape (num_nodes, 3) with the 3D
            coordinates of each node.
        force (torch.Tensor): Tensor of shape (num_edges) with the axial
            force in each edge.
        edge_index (torch.Tensor): Tensor of shape (2, num_edges) defining
            edge connections by node indices.
        load (torch.Tensor): Tensor of shape (num_nodes, 3) with the external
            load applied to each node.

    Returns:
        numpy.ndarray: Array of shape (N, 3) with the residual force vector
        at each node.
    """
    return calculate_residual_force(coords, force, edge_index, load).detach().cpu().numpy()


def _edge_style(force, num_edges, lw_constant, lw_scale):
    """
    Package per-edge color, linewidth and linestyle arrays from `force`.

    Args:
        force (torch.Tensor, optional): Tensor of shape (num_edges) with the
            axial force in each edge. If None, every edge is styled as
            untensioned "target geometry".
        num_edges (int): Number of edges; used to size the output arrays when
            `force` is None.
        lw_constant (bool): If True, every edge gets the same linewidth
            (`lw_scale`). If False, linewidth is scaled by ``sqrt(abs(force))``.
        lw_scale (float): Scale factor applied to every edge's linewidth.

    Returns:
        A tuple containing:
            - is_tension (numpy.ndarray or None): Boolean array of shape (E,),
              True where the edge is in tension, or None when `force` was not
              supplied.
            - colors (numpy.ndarray): Array of shape (E,) with one color per edge.
            - linewidths (numpy.ndarray): Array of shape (E,) with one
              linewidth per edge.
            - linestyles (numpy.ndarray): Array of shape (E,) with one
              linestyle per edge ("-" when `force` is given, ":" otherwise).
    """
    color = PLOT_CONFIG['color']
    line_style = PLOT_CONFIG['line_style']

    if force is not None:
        force_np = force.detach().cpu().numpy()
        is_tension = force_np > 0
        colors = np.where(is_tension, color['edge_tension'], color['edge_compression'])
        base = np.ones(num_edges) if lw_constant else np.sqrt(np.abs(force_np))
        return is_tension, colors, base * lw_scale, np.array([line_style['edge_forced']] * num_edges)

    return (
        None,
        np.array([color['edge_untensioned']] * num_edges),
        np.ones(num_edges) * lw_scale,
        np.array([line_style['edge_untensioned']] * num_edges),
    )


def _deck_quads(coords, is_deck_node):
    """
    Build deck quad vertices from the deck node coordinates.

    Consecutive pairs of deck nodes, ordered by x, are treated as the low- and
    high-y rail at that x position; each adjacent pair of rails becomes one quad.

    Args:
        coords (torch.Tensor): Tensor of shape (num_nodes, 3) with the 3D
            coordinates of each node.
        is_deck_node (torch.Tensor): Boolean tensor of shape (num_nodes)
            selecting the deck rail nodes. Must select an even number of
            nodes, in strict low/high pairs per x position.

    Returns:
        list[numpy.ndarray]: One (4, 3) vertex array per quad, ordered
        counter-clockwise as seen from +x toward the origin.

    Raises:
        ValueError: If `is_deck_node` selects an odd number of nodes.
    """
    deck = coords[is_deck_node.view(-1)].detach().cpu().numpy()
    deck = deck[np.argsort(deck[:, 0])]
    if len(deck) % 2 != 0:
        raise ValueError("Expected an even number of deck coordinates (strict pairs).")
    for i in range(0, len(deck), 2):
        if deck[i, 1] > deck[i + 1, 1]:
            deck[[i, i + 1]] = deck[[i + 1, i]]
    pairs = deck.reshape(-1, 2, 3)
    quads = []
    for k in range(pairs.shape[0] - 1):
        left_low, left_high = pairs[k]
        right_low, right_high = pairs[k + 1]
        quads.append([left_low, left_high, right_high, right_low])
    return quads


def _equalize_3d(ax):
    """
    Force an equal-aspect orthographic cube around the current 3D data.

    Reads the axes' current x/y/z limits, expands the smaller two ranges to
    match the largest so a unit cube in data space renders as a visual cube,
    and switches the axes to an orthographic projection.

    Args:
        ax (mpl_toolkits.mplot3d.axes3d.Axes3D): 3D axes to adjust in place.
    """
    limits = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    centers = [np.mean(lim) for lim in limits]
    radius = 0.5 * max(abs(lim[1] - lim[0]) for lim in limits)
    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect([1, 1, 1])
    ax.set_proj_type("ortho")


def _require(value, name, reason):
    """
    Raise if a draw-time input needed by an enabled `show_*` option is missing.

    Centralizes the "required, or fail with a clear message" check that
    :meth:`~torch_structure.mixins.OverrideResolveMixin._resolve_override`
    applies when resolving a field from a data object, for inputs whose
    requiredness instead depends on a `show_*` flag passed to `plot_3d` /
    `plot_xz` (so it can't be decided at resolution time).

    Args:
        value: The value to check, e.g. `load` or `is_support`.
        name (str): Name of `value`, used in the error message.
        reason (str): Human-readable reason it's needed, e.g. "showing supports".

    Returns:
        The unchanged `value`.

    Raises:
        ValueError: If `value` is None.
    """
    if value is None:
        raise ValueError(f"For {reason} {name} is a required input.")
    return value


def _connected_supports(is_support, edge_index_np, row):
    """
    Select support nodes that appear in one side of the edge index.

    Args:
        is_support (torch.Tensor): Boolean tensor of shape (num_nodes)
            indicating which nodes are supports.
        edge_index_np (numpy.ndarray): Array of shape (2, num_edges) with the
            edge connections by node index.
        row (int): Which side of `edge_index_np` to test membership against —
            0 for source nodes, 1 for destination nodes.

    Returns:
        numpy.ndarray: Indices of the support nodes that also appear in
        ``edge_index_np[row]``.
    """
    is_support_np = is_support.detach().cpu().numpy().astype(bool)
    support_idx = np.flatnonzero(is_support_np)
    return support_idx[np.isin(support_idx, np.unique(edge_index_np[row]))]


def plot_3d(
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
    force_scale=PLOT_CONFIG['vector']['force_scale']['3d'],
    lw_scale=PLOT_CONFIG['line_width']['edge_scale'],
    support_marker_size=PLOT_CONFIG['size']['support_marker']['3d'],
    support_marker_offset=PLOT_CONFIG['offset']['support_marker']['3d'],
    title=None,
    ax=None,
    path=None,
    show=False,
    show_edge_indices=False,
    show_deck=False,
    is_deck_node=None,
):
    """
    Plot a structure in 3D. Internal helper used by :class:`torch_structure.plot.Plotter`.

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
        force_scale (float, optional): Scale factor for visualizing force vectors (load and residual). Default is ``PLOT_CONFIG['vector']['force_scale']['3d']``.
        lw_scale (float, optional): Scale factor for line widths of edges. Default is ``PLOT_CONFIG['line_width']['edge_scale']``.
        support_marker_size (int, optional): Marker size for support symbols. Default is ``PLOT_CONFIG['size']['support_marker']['3d']``.
        support_marker_offset (float, optional): Vertical offset for support markers (to avoid overlap with structure). Default is ``PLOT_CONFIG['offset']['support_marker']['3d']``.
        title (str, optional): Title for the plot. Default is None.
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): An existing 3D axis object to plot on. If None, a new one is created.
        path (str, optional): If provided, saves the figure as a PNG to this file path (without extension).
        show (bool, optional): If True, the plot will be displayed with plt.show(). Default is False.
    """
    color = PLOT_CONFIG['color']
    marker = PLOT_CONFIG['marker']
    vector = PLOT_CONFIG['vector']

    force = force.view(-1) if force is not None else force
    is_support = is_support.view(-1) if is_support is not None else is_support

    # --- package geometry as plain numpy in plot space --------------------
    coords_np = coords.detach().cpu().numpy()
    edge_index_np = edge_index.detach().cpu().numpy()
    num_edges = edge_index_np.shape[1]

    is_tension, edge_colors, edge_lw, _ = _edge_style(
        force, num_edges, lw_constant, lw_scale
    )

    if ax is None:
        fig = plt.figure(figsize=PLOT_CONFIG['figure']['figsize'])
        ax = fig.add_subplot(111, projection="3d")

    legend_handles = []

    # --- supports (drawn before edges, matching the legacy draw order) ----
    if show_supports:
        _require(is_support, "is_support", "showing supports")
        support_pts = coords_np[_connected_supports(is_support, edge_index_np, row=0)].copy()
        support_pts[:, 2] -= support_marker_offset
        draw_node(ax, support_pts, marker=marker['node_support'],
                  size=support_marker_size, color=color['node_support'])

    # --- deck faces -----------------------------------------------------
    if show_deck:
        _require(is_deck_node, "is_deck_node", "showing the deck")
        draw_face(
            ax,
            _deck_quads(coords, is_deck_node),
            facecolor=color['face_deck'],
            edgecolor=color['face_deck_edge'],
            linewidth=PLOT_CONFIG['line_width']['face_deck_edge'],
            alpha=PLOT_CONFIG['alpha']['face_deck'],
        )

    # --- edges --------------------------------------------------------
    segments = coords_np[edge_index_np.T]  # (E, 2, 3)
    draw_edge(ax, segments, colors=edge_colors, linewidths=edge_lw)
    if coords_np.size:
        ax.auto_scale_xyz(coords_np[:, 0], coords_np[:, 1], coords_np[:, 2])

    if is_tension is None:
        legend_handles.append(
            Line2D([], [], color=color['edge_untensioned'], label="Target Geometry")
        )
    else:
        if is_tension.any():
            legend_handles.append(
                Line2D([], [], color=color['edge_tension'],
                       label="Tension (Predicted Equilibrium Geometry)")
            )
        if (~is_tension).any():
            legend_handles.append(
                Line2D([], [], color=color['edge_compression'],
                       label="Compression (Predicted Equilibrium Geometry)")
            )

    # --- edge index labels -----------------------------------------
    if show_edge_indices:
        draw_text(
            ax,
            segments.mean(axis=1),
            [str(i) for i in range(num_edges)],
            color=color['text_edge_index'],
            fontsize=PLOT_CONFIG['size']['edge_index_fontsize'],
        )

    # --- external load arrows ------------------------------------
    if show_load:
        _require(load, "load", "showing external forces")
        load_np = load.detach().cpu().numpy()
        draw_arrow(
            ax,
            coords_np,
            force_scale * load_np,
            color=color['vector_load'],
            length=vector['load_arrow_length_3d'],
            arrow_length_ratio=vector['load_arrow_head_ratio_3d'],
        )

    # --- residual force arrows ---------------------------------
    if show_residual_forces:
        _require(load, "load", "showing residual forces")
        residual = _residual_forces(coords, force, edge_index, load)
        draw_arrow(ax, coords_np, force_scale * residual, color=color['vector_residual'])
        legend_handles.append(
            Line2D([], [], color=color['vector_residual'], label="Residual Force")
        )

    # --- chrome ------------------------------------------------
    if not show_axes:
        ax.set_axis_off()

    if equal_axes:
        _equalize_3d(ax)

    if legend and legend_handles:
        ax.legend(handles=legend_handles, frameon=PLOT_CONFIG['legend']['frameon'])

    if title is not None:
        ax.set_title(title)

    if path is not None:
        ax.get_figure().savefig(
            f"{path}{PLOT_CONFIG['save']['file_format']}",
            bbox_inches=PLOT_CONFIG['save']['bbox_inches'],
        )

    if show:
        plt.show()


def plot_xz(
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
    force_scale=PLOT_CONFIG['vector']['force_scale']['xz'],
    lw_scale=PLOT_CONFIG['line_width']['edge_scale'],
    support_marker_size=PLOT_CONFIG['size']['support_marker']['xz'],
    support_marker_offset=PLOT_CONFIG['offset']['support_marker']['xz'],
    title=None,
    ax=None,
    path=None,
    show=False,
    support_marker_color=PLOT_CONFIG['color']['node_support'],
    highlight_nodes=None,
):
    """
    Plot a structure projected onto the XZ plane. Internal helper used by
    :class:`torch_structure.plot.Plotter`.

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
        force_scale (float, optional): Scale factor for visualizing force vectors (load and residual). Default is ``PLOT_CONFIG['vector']['force_scale']['xz']``.
        lw_scale (float, optional): Scale factor for line widths of edges. Default is ``PLOT_CONFIG['line_width']['edge_scale']``.
        support_marker_size (int, optional): Marker size for support symbols. Default is ``PLOT_CONFIG['size']['support_marker']['xz']``.
        support_marker_offset (float, optional): Vertical offset for support markers (to avoid overlap with structure). Default is ``PLOT_CONFIG['offset']['support_marker']['xz']``.
        title (str, optional): Title for the plot. Default is None.
        ax (matplotlib.axes.Axes, optional): An existing 2D axis object to plot on. If None, a new one is created.
        path (str, optional): If provided, saves the figure as a PNG to this file path (without extension).
        show (bool, optional): If True, the plot will be displayed with plt.show(). Default is False.
        support_marker_color (str, optional): Colour of the support markers. Default is ``PLOT_CONFIG['color']['node_support']``.
        highlight_nodes (Sequence[int], optional): Node indices to mark as considered in the current step.
    """
    color = PLOT_CONFIG['color']
    marker = PLOT_CONFIG['marker']
    vector = PLOT_CONFIG['vector']
    legend_cfg = PLOT_CONFIG['legend']

    force = force.view(-1) if force is not None else force
    is_support = is_support.view(-1) if is_support is not None else is_support

    # --- package geometry as plain numpy in plot space (x, z) ------------
    coords_np = coords.detach().cpu().numpy()
    xz = coords_np[:, [0, 2]]
    num_nodes = coords_np.shape[0]
    edge_index_np = edge_index.detach().cpu().numpy()
    num_edges = edge_index_np.shape[1]

    is_tension, edge_colors, edge_lw, edge_ls = _edge_style(
        force, num_edges, lw_constant, lw_scale
    )
    geometry_label = "Converged Geometry" if is_tension is None else None

    if ax is None:
        fig = plt.figure(figsize=PLOT_CONFIG['figure']['figsize'])
        ax = fig.add_subplot(111)

    legend_handles = []

    # --- edges: collapse reciprocal pairs, keep first occurrence --------
    lo = np.minimum(edge_index_np[0], edge_index_np[1]).astype(np.int64)
    hi = np.maximum(edge_index_np[0], edge_index_np[1]).astype(np.int64)
    _, keep = np.unique(lo * max(num_nodes, 1) + hi, return_index=True)
    keep.sort()
    draw_edge(
        ax,
        xz[edge_index_np.T][keep],
        colors=edge_colors[keep],
        linewidths=edge_lw[keep],
        linestyles=list(edge_ls[keep]),
    )
    ax.autoscale_view()

    if is_tension is None:
        legend_handles.append(
            Line2D([], [], color=color['edge_untensioned'],
                   linestyle=PLOT_CONFIG['line_style']['edge_untensioned'], label=geometry_label)
        )
    else:
        if is_tension.any():
            legend_handles.append(
                Line2D([], [], color=color['edge_tension'], label="Tension")
            )
        if (~is_tension).any():
            legend_handles.append(
                Line2D([], [], color=color['edge_compression'], label="Compression")
            )

    # --- external load arrows ----------------------------------
    if show_load:
        _require(load, "load", "showing external forces")
        load_np = load.detach().cpu().numpy()
        vectors = force_scale * load_np[:, [0, 2]]
        moving = np.linalg.norm(vectors, axis=1) > vector['xz_load_min_magnitude']
        if moving.any():
            draw_arrow(ax, xz[moving], vectors[moving], color=color['vector_load'])
            legend_handles.append(
                Line2D([], [], color=color['vector_load'], label="External Load")
            )

    # --- supports (drawn after edges/loads, matching legacy draw order) -
    if show_supports:
        _require(is_support, "is_support", "showing supports")
        support_pts = xz[_connected_supports(is_support, edge_index_np, row=1)].copy()
        support_pts[:, 1] -= support_marker_offset
        draw_node(
            ax, support_pts, marker=marker['node_support'],
            size=support_marker_size, color=support_marker_color
        )
        legend_handles.append(
            Line2D([], [], color=support_marker_color, marker=marker['node_support'],
                   linestyle="None", label="Support")
        )

    # --- residual force arrows -------------------------------
    if show_residual_forces:
        _require(load, "load", "showing residual forces")
        residual = _residual_forces(coords, force, edge_index, load)
        draw_arrow(ax, xz, force_scale * residual[:, [0, 2]], color=color['vector_residual'])
        legend_handles.append(
            Line2D([], [], color=color['vector_residual'], label="Residual Force")
        )

    # --- highlighted nodes --------------------------------
    if highlight_nodes is not None:
        draw_node(ax, xz[list(highlight_nodes)], marker=marker['node_highlight'],
                  size=PLOT_CONFIG['size']['node_highlight_marker'], color=color['node_highlight'])
        legend_handles.append(
            Line2D([], [], color=color['node_highlight'], marker=marker['node_highlight'],
                   linestyle="None", label="Considered in Step")
        )

    # --- chrome -------------------------------------------
    if not show_axes:
        ax.set_axis_off()

    if equal_axes:
        ax.set_aspect("equal", adjustable="box")

    if legend and legend_handles:
        ax.legend(
            handles=legend_handles,
            frameon=legend_cfg['frameon'],
            ncol=legend_cfg['ncol_xz'],
            loc=legend_cfg['loc_xz'],
            bbox_to_anchor=legend_cfg['bbox_to_anchor_xz'],
        )

    if title is not None:
        ax.set_title(title, pad=PLOT_CONFIG['title']['pad_xz'],
                     fontsize=PLOT_CONFIG['size']['title_fontsize_xz'])

    if path is not None:
        ax.get_figure().savefig(
            f"{path}{PLOT_CONFIG['save']['file_format']}",
            bbox_inches=PLOT_CONFIG['save']['bbox_inches'],
        )

    if show:
        plt.show()
