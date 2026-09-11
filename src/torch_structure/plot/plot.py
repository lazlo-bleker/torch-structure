import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from torch_structure.plot.config import PLOT_CONFIG
from torch_structure.plot.primitives import (
    draw_arrow,
    draw_edge,
    draw_face,
    draw_node,
    draw_text,
)


def _edge_style(force, num_edges, lw_constant, lw_scale):
    """
    Package per-edge color, linewidth and linestyle arrays from `force`.

    Args:
        force (numpy.ndarray, optional): Array of shape (num_edges) with the
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
        is_tension = force > 0
        colors = np.where(is_tension, color['edge_tension'], color['edge_compression'])
        base = np.ones(num_edges) if lw_constant else np.sqrt(np.abs(force))
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
        coords (numpy.ndarray): Array of shape (num_nodes, 3) with the 3D
            coordinates of each node.
        is_deck_node (numpy.ndarray): Boolean array of shape (num_nodes)
            selecting the deck rail nodes. Must select an even number of
            nodes, in strict low/high pairs per x position.

    Returns:
        list[numpy.ndarray]: One (4, 3) vertex array per quad, ordered
        counter-clockwise as seen from +x toward the origin.

    Raises:
        ValueError: If `is_deck_node` selects an odd number of nodes.
    """
    deck = coords[is_deck_node.reshape(-1)]
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


def _equalize_3d(ax, zoom=1.0):
    """
    Force an equal-aspect orthographic cube around the current 3D data.

    Reads the axes' current x/y/z limits, expands the smaller two ranges to
    match the largest so a unit cube in data space renders as a visual cube,
    and switches the axes to an orthographic projection.

    Args:
        ax (mpl_toolkits.mplot3d.axes3d.Axes3D): 3D axes to adjust in place.
        zoom (float, optional): >= 1.0. 1.0 (default) keeps the tight,
            unzoomed cube.
    """
    limits = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]
    centers = [np.mean(lim) for lim in limits]
    radius = 0.5 * max(abs(lim[1] - lim[0]) for lim in limits) / zoom
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
        is_support (numpy.ndarray): Boolean array of shape (num_nodes)
            indicating which nodes are supports.
        edge_index_np (numpy.ndarray): Array of shape (2, num_edges) with the
            edge connections by node index.
        row (int): Which side of `edge_index_np` to test membership against —
            0 for source nodes, 1 for destination nodes.

    Returns:
        numpy.ndarray: Indices of the support nodes that also appear in
        ``edge_index_np[row]``.
    """
    support_idx = np.flatnonzero(is_support.astype(bool))
    return support_idx[np.isin(support_idx, np.unique(edge_index_np[row]))]


def plot_inset_axis_3d(fig, ax):
    """
    Draw a small XYZ orientation triad pinned to the bottom-left corner of
    the figure, independent of `ax`'s own position/data limits, that stays
    in sync with `ax` as it's rotated interactively.

    Args:
        fig (matplotlib.figure.Figure): Figure to attach the inset axes to.
        ax (mpl_toolkits.mplot3d.axes3d.Axes3D): Main 3D axes whose view
            angle (elev/azim/roll) the inset triad tracks.

    Returns:
        mpl_toolkits.mplot3d.axes3d.Axes3D: The inset axes.
    """
    cfg = PLOT_CONFIG['inset_axis']
    color = cfg['color']

    inset_ax = fig.add_axes(cfg['rect'], projection="3d")
    inset_ax.disable_mouse_rotation()  # never rotate/pan/zoom from its own mouse events
    inset_ax.set_navigate(False)       # excluded from toolbar pan/zoom too
    inset_ax.set_facecolor("none")
    inset_ax.set_xlim3d(-1, 1)
    inset_ax.set_ylim3d(-1, 1)
    inset_ax.set_zlim3d(-1, 1)
    inset_ax.set_box_aspect([1, 1, 1])
    inset_ax.set_axis_off()

    axes_vectors = np.eye(3)
    draw_arrow(inset_ax, np.zeros((3, 3)), axes_vectors,
               color=color, arrow_length_ratio=cfg['arrow_length_ratio'],
               linewidth=cfg['linewidth'])
    draw_text(inset_ax, axes_vectors * cfg['label_offset'], ["X", "Y", "Z"],
              color=color, fontsize=cfg['fontsize'])

    def _sync_rotation(event=None):
        if (inset_ax.elev, inset_ax.azim) != (ax.elev, ax.azim):
            inset_ax.view_init(elev=ax.elev, azim=ax.azim, roll=ax.roll)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _sync_rotation)
    _sync_rotation()
    return inset_ax


def plot_inset_axis_xz(fig):
    """
    Draw a small XZ orientation indicator pinned to the bottom-left corner
    of the figure, independent of the main axes' position/data limits.
    Static — `plot_xz`'s view never rotates, so there's nothing to track.

    Args:
        fig (matplotlib.figure.Figure): Figure to attach the inset axes to.

    Returns:
        matplotlib.axes.Axes: The inset axes.
    """
    cfg = PLOT_CONFIG['inset_axis']
    color = cfg['color']

    inset_ax = fig.add_axes(cfg['rect'])
    inset_ax.set_navigate(False)  # excluded from toolbar pan/zoom
    inset_ax.set_facecolor("none")
    inset_ax.set_xlim(-1, 1)
    inset_ax.set_ylim(-1, 1)
    inset_ax.set_aspect("equal")
    inset_ax.set_axis_off()

    axes_vectors = np.eye(2)
    draw_arrow(inset_ax, np.zeros((2, 2)), axes_vectors, color=color, linewidth=cfg['linewidth'])
    draw_text(inset_ax, axes_vectors * cfg['label_offset'], ["X", "Z"],
              color=color, fontsize=cfg['fontsize'])
    return inset_ax


def plot_3d(plot_data, title=None, legend=True):
    """
    Plot a structure in 3D. Internal helper used by :class:`torch_structure.plot.Plotter`.

    Args:
        plot_data (dict): Drawing inputs collected by
            :meth:`~torch_structure.plot.Plotter._collect_plot_data` — plain
            numpy arrays `coords` (N, 3), `edge_index` (2, E), and optional
            `is_support` (N,), `force` (E,), `load` (N, 3),
            `residual_force` (N, 3, precomputed) and `is_deck_node` (N,).
        title (str, optional): Title for the plot. Default is None.
        legend (bool, optional): If True, a legend will be displayed. Default is True.

    Every other drawing setting (`show_supports`, `show_load`,
    `show_residual_forces`, `show_axes`, `equal_axes`, `lw_constant`,
    `force_scale`, `lw_scale`, `support_marker_size`,
    `support_marker_offset`, `show_edge_indices`, `show_deck`) is read from
    :data:`torch_structure.plot.config.PLOT_CONFIG`.
    """
    color = PLOT_CONFIG['color']
    marker = PLOT_CONFIG['marker']
    vector = PLOT_CONFIG['vector']
    show = PLOT_CONFIG['show']

    coords_np = plot_data['coords']
    edge_index_np = plot_data['edge_index']
    is_support = plot_data.get('is_support')
    force = plot_data.get('force')
    load = plot_data.get('load')
    residual_force = plot_data.get('residual_force')
    is_deck_node = plot_data.get('is_deck_node')

    force_scale = vector['force_scale']['3d']
    lw_scale = PLOT_CONFIG['line_width']['edge_scale']
    support_marker_size = PLOT_CONFIG['size']['support_marker']['3d']
    support_marker_offset = PLOT_CONFIG['offset']['support_marker']['3d']

    num_edges = edge_index_np.shape[1]

    is_tension, edge_colors, edge_lw, _ = _edge_style(
        force, num_edges, show['lw_constant'], lw_scale
    )

    fig = plt.figure(figsize=PLOT_CONFIG['figure']['figsize'])
    ax = fig.add_subplot(111, projection="3d")

    legend_handles = []

    # --- supports (drawn before edges, matching the legacy draw order) ----
    if show['show_supports']:
        _require(is_support, "is_support", "showing supports")
        support_pts = coords_np[_connected_supports(is_support, edge_index_np, row=0)].copy()
        support_pts[:, 2] -= support_marker_offset
        draw_node(ax, support_pts, marker=marker['node_support'],
                  size=support_marker_size, color=color['node_support'])

    # --- deck faces -----------------------------------------------------
    if show['show_deck']:
        _require(is_deck_node, "is_deck_node", "showing the deck")
        draw_face(
            ax,
            _deck_quads(coords_np, is_deck_node),
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
    if show['show_edge_indices']:
        draw_text(
            ax,
            segments.mean(axis=1),
            [str(i) for i in range(num_edges)],
            color=color['text_edge_index'],
            fontsize=PLOT_CONFIG['size']['edge_index_fontsize'],
        )

    # --- external load arrows ------------------------------------
    if show['show_load']:
        _require(load, "load", "showing external forces")
        draw_arrow(
            ax,
            coords_np,
            force_scale * load,
            color=color['vector_load'],
            length=vector['load_arrow_length_3d'],
            arrow_length_ratio=vector['load_arrow_head_ratio_3d'],
        )

    # --- residual force arrows ---------------------------------
    if show['show_residual_forces']:
        _require(residual_force, "load", "showing residual forces")
        draw_arrow(ax, coords_np, force_scale * residual_force, color=color['vector_residual'])
        legend_handles.append(
            Line2D([], [], color=color['vector_residual'], label="Residual Force")
        )

    # --- chrome ------------------------------------------------
    if not show['show_axes']:
        ax.set_axis_off()

    if show['equal_axes']:
        _equalize_3d(ax, zoom=PLOT_CONFIG['figure']['zoom_3d'])

    if show['show_inset_axis']:
        plot_inset_axis_3d(fig, ax)

    if legend and legend_handles:
        ax.legend(handles=legend_handles, frameon=PLOT_CONFIG['legend']['frameon'])

    if title is not None:
        ax.set_title(title)


def plot_xz(plot_data, title=None, legend=True):
    """
    Plot a structure projected onto the XZ plane. Internal helper used by
    :class:`torch_structure.plot.Plotter`.

    Args:
        plot_data (dict): Drawing inputs collected by
            :meth:`~torch_structure.plot.Plotter._collect_plot_data` — plain
            numpy arrays `coords` (N, 3), `edge_index` (2, E), and optional
            `is_support` (N,), `force` (E,), `load` (N, 3),
            `residual_force` (N, 3, precomputed) and `highlight_nodes`
            (a sequence of node indices).
        title (str, optional): Title for the plot. Default is None.
        legend (bool, optional): If True, a legend will be displayed. Default is True.

    Every other drawing setting (`show_supports`, `show_load`,
    `show_residual_forces`, `show_axes`, `equal_axes`, `lw_constant`,
    `force_scale`, `lw_scale`, `support_marker_size`,
    `support_marker_offset`) is read from
    :data:`torch_structure.plot.config.PLOT_CONFIG`.
    """
    color = PLOT_CONFIG['color']
    marker = PLOT_CONFIG['marker']
    vector = PLOT_CONFIG['vector']
    legend_cfg = PLOT_CONFIG['legend']
    show = PLOT_CONFIG['show']

    coords_np = plot_data['coords']
    edge_index_np = plot_data['edge_index']
    is_support = plot_data.get('is_support')
    force = plot_data.get('force')
    load = plot_data.get('load')
    residual_force = plot_data.get('residual_force')
    highlight_nodes = plot_data.get('highlight_nodes')

    force_scale = vector['force_scale']['xz']
    lw_scale = PLOT_CONFIG['line_width']['edge_scale']
    support_marker_size = PLOT_CONFIG['size']['support_marker']['xz']
    support_marker_offset = PLOT_CONFIG['offset']['support_marker']['xz']
    support_marker_color = color['node_support']

    # --- package geometry in plot space (x, z) ---------------------------
    xz = coords_np[:, [0, 2]]
    num_nodes = coords_np.shape[0]
    num_edges = edge_index_np.shape[1]

    is_tension, edge_colors, edge_lw, edge_ls = _edge_style(
        force, num_edges, show['lw_constant'], lw_scale
    )
    geometry_label = "Converged Geometry" if is_tension is None else None

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
    if show['show_load']:
        _require(load, "load", "showing external forces")
        vectors = force_scale * load[:, [0, 2]]
        moving = np.linalg.norm(vectors, axis=1) > vector['xz_load_min_magnitude']
        if moving.any():
            draw_arrow(ax, xz[moving], vectors[moving], color=color['vector_load'])
            legend_handles.append(
                Line2D([], [], color=color['vector_load'], label="External Load")
            )

    # --- supports (drawn after edges/loads, matching legacy draw order) -
    if show['show_supports']:
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
    if show['show_residual_forces']:
        _require(residual_force, "load", "showing residual forces")
        draw_arrow(ax, xz, force_scale * residual_force[:, [0, 2]], color=color['vector_residual'])
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
    if not show['show_axes']:
        ax.set_axis_off()

    if show['equal_axes']:
        ax.set_aspect("equal", adjustable="box")

    if show['show_inset_axis']:
        plot_inset_axis_xz(fig)

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
