"""
A general plotter for structures: nodes and edges in 3D or in a 2D projection, each colored, sized
and labelled by any node or edge attribute, together with supports, node vectors such as loads, a
reference geometry, and the pattern elements of registered patterns.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, is_color_like, to_rgba
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


TENSION_COLOR = "#E40714"
COMPRESSION_COLOR = "#0578BF"
LOAD_COLOR = "#007F00"
REACTION_COLOR = "#E7461E"

_EDGE_COLOR = "#7d7c78"
_NODE_COLOR = "#3d3c39"
_ZERO_COLOR = "#b5b4ae"
_GHOST_COLOR = "#9a9a9a"
_HIGHLIGHT_COLOR = "#E7461E"
_VECTOR_COLORS = ["#8E44AD", "#16A085", "#2C3E50", "#B7950B", "#C0392B"]

_VIEWS = {"3d": (0, 1, 2), "xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
_MAX_CATEGORIES = 10


def _numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype in (torch.bfloat16, torch.float16):
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def _attr(data, name):
    value = getattr(data, name, None)
    if value is None:
        raise ValueError(f"'{type(data).__name__}' has no attribute '{name}' to plot.")
    return value


def _resolve_coords(data, coords):
    if coords is None:
        coords = "coords"
    if isinstance(coords, str):
        coords = _attr(data, coords)
    coords = _numpy(coords).astype(np.float64)
    coords = coords.reshape(len(coords), -1) if coords.size else np.zeros((len(coords), 3))
    if coords.shape[1] < 3:
        coords = np.hstack([coords, np.zeros((len(coords), 3 - coords.shape[1]))])
    return coords


def _canonical_edges(data, num_edges):
    # one of the two directed edges that make up every undirected edge, where the graph says which
    directed = getattr(data, "directed_mask", None)
    if directed is not None and directed.numel() == num_edges:
        return _numpy(directed).reshape(-1).astype(bool)
    return np.ones(num_edges, dtype=bool)


def _mask(mask, count, what):
    if mask is None:
        return np.ones(count, dtype=bool)
    if isinstance(mask, (bool, np.bool_)):
        return np.full(count, bool(mask))
    mask = _numpy(mask).reshape(-1)
    if mask.dtype != bool:
        # indices instead of a boolean mask
        indices, mask = mask.astype(np.int64), np.zeros(count, dtype=bool)
        mask[indices] = True
    if len(mask) != count:
        raise ValueError(f"The {what} mask has {len(mask)} entries, expected {count}.")
    return mask


class _Edges:
    # the edges drawn: their rows in edge_index, and how to read a per-edge value for them

    def __init__(self, data, edge_mask):
        edge_index = getattr(data, "edge_index", None)
        self.edge_index = np.zeros((2, 0), dtype=np.int64) if edge_index is None else _numpy(edge_index).astype(np.int64)
        self.num_edges = self.edge_index.shape[1]
        self.canonical = _canonical_edges(data, self.num_edges)
        self.position = np.cumsum(self.canonical) - 1

        # a mask may be given for all edges or for the canonical ones only
        if edge_mask is not None and not isinstance(edge_mask, (bool, np.bool_)):
            mask = _numpy(edge_mask).reshape(-1)
            if mask.dtype == bool and len(mask) == self.canonical.sum() != self.num_edges:
                full = np.zeros(self.num_edges, dtype=bool)
                full[self.canonical] = mask
                edge_mask = full
        self.ids = np.flatnonzero(self.canonical & _mask(edge_mask, self.num_edges, "edge"))

    def values(self, value):
        value = _numpy(value)
        value = value.reshape(len(value), -1) if value.ndim > 1 else value.reshape(-1, 1)
        value = value[:, 0] if value.shape[1] == 1 else value
        if len(value) == self.num_edges:
            return value[self.ids]
        if len(value) == self.canonical.sum():
            return value[self.position[self.ids]]
        raise ValueError(
            f"An edge value has {len(value)} entries, expected one per edge ({self.num_edges}) "
            f"or one per undirected edge ({int(self.canonical.sum())})."
        )


def _lookup(data, value, values, special=()):
    # an attribute name, or a tensor or array of values; returns the values for the drawn items and a name
    if isinstance(value, str):
        if value in special:
            return special[value], value
        return values(_attr(data, value)), value
    return values(value), None


def _color_values(data, value, values, default, name_hint=None):
    # constant color, or values to map to colors
    if value is None:
        return default, None, None
    if isinstance(value, str) and getattr(data, value, None) is None and is_color_like(value):
        return value, None, None
    if isinstance(value, tuple) and is_color_like(value):
        return value, None, None
    resolved, name = _lookup(data, value, values)
    return None, resolved, name or name_hint


def _map_colors(values, name, cmap, clim, default_color):
    """Turns values into colors: categories for booleans and a few integers, a colormap otherwise.

    Returns the RGBA colors, legend entries as ``(color, label)`` pairs, and a mappable for a
    colorbar, or None.
    """
    values = np.asarray(values)
    label = name or "value"

    if values.dtype == bool:
        colors = np.where(values[:, None], to_rgba(_HIGHLIGHT_COLOR), to_rgba(default_color))
        return colors, [(_HIGHLIGHT_COLOR, label), (default_color, f"not {label}")], None

    finite = values[np.isfinite(values)] if values.dtype.kind == "f" else values
    categories = np.unique(finite)
    if values.dtype.kind in "iu" and len(categories) <= _MAX_CATEGORIES:
        palette = colormaps["tab10"]
        lookup = {v: palette(i % 10) for i, v in enumerate(categories.tolist())}
        colors = np.array([lookup[v] for v in values.tolist()]).reshape(-1, 4)
        return colors, [(lookup[v], f"{label} = {v}") for v in categories.tolist()], None

    if clim is not None:
        low, high = clim
    elif len(finite) == 0:
        low, high = 0.0, 1.0
    elif finite.min() < 0 < finite.max():
        # values of both signs: a diverging map centred on zero, red for positive like tension
        high = float(np.abs(finite).max())
        low = -high
    else:
        low, high = float(finite.min()), float(finite.max())
    if cmap is None:
        cmap = "RdBu_r" if low < 0 < high else "viridis"
    norm = Normalize(low, high if high > low else low + 1.0)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    colors = mappable.to_rgba(values.astype(np.float64))
    colors[~np.isfinite(values.astype(np.float64))] = to_rgba(_ZERO_COLOR)
    mappable.set_array(values)
    mappable._ts_label = label
    return colors, [], mappable


def _widths(values, lw_scale, minimum=0.5, maximum=3.0):
    # line widths grow with the square root of the magnitude, the largest at `maximum`
    magnitude = np.abs(np.asarray(values, dtype=np.float64))
    top = np.nanmax(magnitude) if magnitude.size else 0.0
    relative = np.sqrt(magnitude / top) if top > 0 else np.zeros_like(magnitude)
    return lw_scale * (minimum + (maximum - minimum) * np.nan_to_num(relative))


def _format(value, label_format):
    if isinstance(value, str):
        return value
    value = np.asarray(value)
    if value.ndim > 0:
        return "(" + ", ".join(_format(v, label_format) for v in value.tolist()) + ")"
    value = value.item()
    if isinstance(value, (bool, int)):
        return str(value)
    return label_format.format(value)


def _node_names(data, ids):
    from torch_structure.data.data import decode

    names = _numpy(_attr(data, "name")).reshape(-1)
    out = []
    for i in ids:
        try:
            out.append(decode(int(names[i])) if names[i] > 0 else "")
        except (OverflowError, UnicodeDecodeError):
            out.append(str(names[i]))
    return out


def _labels(data, value, values, label_format, special):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) and isinstance(value[0], str):
        value = np.array(value, dtype=object)
    resolved, _ = _lookup(data, value, values, special)
    return [_format(v, label_format) for v in resolved]


def _supports(data, supports):
    # the supported nodes, and which of them also hold rotations; None if there is nothing to draw
    if supports is False:
        return None
    try:
        is_support = _numpy(data.is_support).reshape(-1).astype(bool)
    except AttributeError:
        if supports:
            raise ValueError("Plotting supports requires an 'is_support' or 'support_xyz' attribute.") from None
        return None
    try:
        fixed = _numpy(data.support_rxyz).astype(bool).any(axis=1)
    except AttributeError:
        fixed = np.zeros_like(is_support)
    return is_support, fixed


def _vector_sets(data, loads, reactions, vectors, num_nodes, node_ids):
    # (label, vectors, color, whether the arrows stay outside the structure) for each set of node vectors to draw
    def node_vectors(value, default):
        value = _attr(data, default) if value is True else _attr(data, value) if isinstance(value, str) else value
        value = _numpy(value)
        if value.shape != (num_nodes, 3):
            raise ValueError(f"Node vectors need the shape [num_nodes, 3] = [{num_nodes}, 3], got {list(value.shape)}.")
        return value[node_ids]

    sets = []
    if loads is not None and loads is not False:
        sets.append(("Load", node_vectors(loads, "load"), LOAD_COLOR, False))
    if reactions is not None and reactions is not False:
        sets.append(("Reaction", node_vectors(reactions, "reaction_force"), REACTION_COLOR, True))
    for i, (label, value) in enumerate((vectors or {}).items()):
        sets.append((label, node_vectors(value, None), _VECTOR_COLORS[i % len(_VECTOR_COLORS)], False))
    return sets


def _inward_directions(xyz, edge_index):
    # at each node, the summed unit directions along its members: the way into the structure
    inward = np.zeros_like(xyz)
    if edge_index.shape[1]:
        src, dst = edge_index
        direction = xyz[dst] - xyz[src]
        length = np.linalg.norm(direction, axis=1, keepdims=True)
        np.add.at(inward, src, direction / np.where(length > 0, length, 1.0))
    return inward


def _ghost(data, ghost):
    # the coordinates and segments of a reference geometry: a graph of its own, or other coordinates of this one
    if ghost is None:
        return None
    if hasattr(ghost, "edge_index"):
        coords = _resolve_coords(ghost, None)
        edges = _Edges(ghost, None)
    else:
        coords = _resolve_coords(data, ghost)
        edges = _Edges(data, None)
    return coords, edges.edge_index[:, edges.ids]


def _set_limits(ax, points, dims, equal_axes):
    # the limits hold everything drawn on the axis so far, also by earlier calls on the same axis
    points = np.concatenate([np.asarray(p, dtype=float).reshape(-1, 3) for p in points], axis=0)[:, dims]
    points = points[np.isfinite(points).all(axis=1)]
    previous = getattr(ax, "_ts_extent", None)
    if previous is not None:
        points = np.concatenate([points, previous], axis=0)
    if not len(points):
        return
    low, high = points.min(axis=0), points.max(axis=0)
    ax._ts_extent = np.stack([low, high])
    mid, half = (low + high) / 2, np.maximum((high - low) / 2, 1e-9)

    if len(dims) == 3:
        if equal_axes:
            # one scale on all axes, with a flat structure still shown with some depth
            half = np.maximum(half, 0.15 * half.max())
            ax.set_box_aspect(tuple(half))
            ax.set_proj_type("ortho")
        half = half * 1.02
        ax.set_xlim3d(mid[0] - half[0], mid[0] + half[0])
        ax.set_ylim3d(mid[1] - half[1], mid[1] + half[1])
        ax.set_zlim3d(mid[2] - half[2], mid[2] + half[2])
    else:
        half = half + 0.05 * half.max()
        ax.set_xlim(mid[0] - half[0], mid[0] + half[0])
        ax.set_ylim(mid[1] - half[1], mid[1] + half[1])
        if equal_axes:
            ax.set_aspect("equal", adjustable="box")


def plot_structure(
    data,
    coords=None,
    view="3d",
    edge_color=None,
    edge_width=None,
    edge_labels=None,
    edge_mask=None,
    node_color=None,
    node_size=None,
    node_labels=None,
    node_mask=None,
    supports=None,
    loads=None,
    reactions=None,
    vectors=None,
    vector_scale=None,
    ghost=None,
    ghost_label="Reference",
    pattern=None,
    cmap=None,
    clim=None,
    colorbar=True,
    lw_scale=1.0,
    label_format="{:.3g}",
    fontsize=8,
    legend=True,
    title=None,
    show_axes=False,
    equal_axes=True,
    ax=None,
    path=None,
    show=False,
):
    """
    Plots a structure, in 3D or in a 2D projection, with its edges and nodes colored, sized and
    labelled by any of their attributes.

    The edge and node arguments accept the same kinds of value:

    - the name of an edge or node attribute, e.g. ``edge_color="force"`` or
      ``node_labels="name"``;
    - a tensor or array of values, one per node, or one per edge (all directed edges, or one per
      undirected edge, as e.g. ``data.force[data.directed_mask]``);
    - a constant, e.g. ``edge_color="black"`` or ``node_size=20``.

    Values are mapped to colors by their type: booleans highlight the ``True`` entries, integers
    with at most 10 distinct values get one color each, e.g. a ``material_id``, and all other
    numbers a colormap with a colorbar, diverging around zero when values of both signs occur.

    Every undirected edge is drawn once, along its ``directed_mask`` direction.

    Args:
        data (StructData): the structure to plot.
        coords (torch.Tensor | str, optional): node coordinates of shape [num_nodes, 3], or the
            name of the node attribute holding them, e.g. a form-found geometry. Defaults to
            ``data.coords``.
        view (str): ``"3d"``, or the plane of a 2D projection: ``"xy"`` (plan), ``"xz"`` or
            ``"yz"`` (elevations). Default is ``"3d"``.
        edge_color (str | torch.Tensor, optional): edge colors, see above. By default, edges
            with a ``force`` attribute are red in tension and blue in compression, and gray
            otherwise.
        edge_width (str | torch.Tensor | float, optional): edge line widths: a constant, or values
            whose magnitude sets the width, from ``0.5`` to ``3`` times ``lw_scale``. By default,
            the magnitude of ``force`` if the structure has one, and ``1.5`` otherwise.
        edge_labels (str | torch.Tensor | list[str], optional): text at each edge's midpoint:
            ``"index"`` for its row in ``edge_index``, an attribute name, values, or strings.
        edge_mask (torch.Tensor | bool, optional): which edges to draw: a boolean mask or edge
            indices, for all edges or for the undirected ones. ``False`` draws none.
        node_color (str | torch.Tensor, optional): node colors, see above. Nodes are only drawn
            when ``node_color`` or ``node_size`` is given, or with ``pattern``.
        node_size (str | torch.Tensor | float, optional): marker sizes: a constant, or values
            whose magnitude sets the size. Default is 12 when nodes are drawn.
        node_labels (str | torch.Tensor | list[str], optional): text at each node: ``"index"``,
            ``"name"``, an attribute name, values, or strings. Values with several components,
            e.g. ``"coords"``, are written as tuples.
        node_mask (torch.Tensor | bool, optional): which nodes to draw and label, as a boolean
            mask or node indices. Edges are drawn regardless.
        supports (bool, optional): mark the supports, circles for pinned and squares for
            nodes that also hold rotations. By default, drawn if the structure has supports.
        loads (bool | str | torch.Tensor, optional): draw load arrows: ``True`` for
            ``data.load``, an attribute name, or a tensor of shape [num_nodes, 3].
        reactions (bool | str | torch.Tensor, optional): draw the support reactions as arrows
            outside the structure, ending at their support when they push on it and starting
            there when they pull: ``True`` for ``data.reaction_force``, an attribute name, or a
            tensor of shape [num_nodes, 3].
        vectors (dict[str, str | torch.Tensor], optional): further node vectors to draw as
            arrows, by legend label, e.g. ``{"Residual": residual_force}``.
        vector_scale (float, optional): length of the drawn arrow per unit of load, reaction or
            vector, shared by all of them. By default, the longest arrow is 15 % of the
            structure's size.
        ghost (StructData | torch.Tensor | str, optional): a reference geometry drawn dashed
            and gray: another structure, or other coordinates for this one, e.g. a target
            geometry or the coordinates before form finding.
        ghost_label (str): legend label of ``ghost``. Default is ``"Reference"``.
        pattern (bool | str | list[str] | dict, optional): draw the pattern elements of the
            registered patterns, level by level: mirror planes as translucent planes, rotation
            axes as dotted lines with arcs that follow one step of the rotation, and
            translations as arrows from each copy to the next. ``True`` for all patterns, their
            names, or a dict of options, see
            [draw_pattern_elements][torch_structure.plot.pattern.draw_pattern_elements]:

            - ``"patterns"`` (str | list[str]): the patterns to draw. Defaults to all.
            - ``"levels"`` (list[int] | dict[str, list[int]]): the levels to draw, for every
              pattern or per pattern name. Defaults to all.
            - ``"copies"`` (bool): draw each level in every copy the levels around it make of
              it, instead of only in the first. Default is False.
            - ``"show_labels"`` (bool): label each level's element, e.g. ``T1``. Default is
              True.
            - ``"show_seeds"`` (bool): ring the nodes at expansion position 0. Default is
              False.
            - ``"color"``, ``"plane_alpha"``, ``"atol"``: the elements' color, the mirror
              planes' opacity, and the tolerance for reading the pattern matrices.

            For example ``pattern={"patterns": "truss", "levels": [0, 1], "copies": True}``.
            Only in 3D.
        cmap (str, optional): colormap for continuous values. Defaults to ``"viridis"``, or
            ``"RdBu_r"`` for values of both signs.
        clim (tuple[float, float], optional): value range of the colormap. Defaults to the
            range of the values, symmetric around zero for values of both signs.
        colorbar (bool): if True, adds a colorbar for colors from a colormap. Default is True.
        lw_scale (float): scale factor for all edge line widths. Default is 1.0.
        label_format (str): format of numeric labels. Default is ``"{:.3g}"``.
        fontsize (float): font size of the labels. Default is 8.
        legend (bool): if True, shows a legend of what is drawn. Default is True.
        title (str, optional): title of the plot.
        show_axes (bool): if True, axis lines and labels are shown. Default is False.
        equal_axes (bool): if True, all axes share one scale. Default is True.
        ax (matplotlib.axes.Axes, optional): an existing axis to plot on, 3D for ``view="3d"``.
            Plotting several times on one axis overlays the plots. If None, a new figure is
            created.
        path (str, optional): if given, saves the figure to this file; without an extension as
            PNG.
        show (bool): if True, the plot is displayed with ``plt.show()``. Default is False.

    Returns:
        matplotlib.axes.Axes: the axis plotted on.

    Example:
        ```python
        data.plot(edge_color="length", edge_labels="index", node_labels="name")
        data.plot(view="xz", loads=True, reactions=True)
        data.plot(edge_color="material_id", ghost="initial_coords")
        data.plot(pattern=True, node_labels="index")
        ```
    """
    if view not in _VIEWS:
        raise ValueError(f"Unknown view '{view}'; expected one of {list(_VIEWS)}.")
    dims = list(_VIEWS[view])
    is_3d = len(dims) == 3
    if pattern and not is_3d:
        raise ValueError("Pattern elements can only be drawn in the 3D view.")

    xyz = _resolve_coords(data, coords)
    num_nodes = len(xyz)
    edges = _Edges(data, edge_mask)
    node_ids = np.flatnonzero(_mask(node_mask, num_nodes, "node"))

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)
    elif is_3d != (getattr(ax, "name", "") == "3d"):
        raise ValueError(f"view='{view}' needs a {'3D' if is_3d else '2D'} axis.")
    if is_3d:
        # supports, arrows and labels stay on top of the edges instead of being sorted by depth
        ax.computed_zorder = False

    # the extent sets default arrow lengths and pattern element sizes
    extent = float(np.linalg.norm(np.ptp(xyz, axis=0))) if num_nodes else 1.0
    extent = extent or 1.0

    legend_entries = list(getattr(ax, "_ts_legend", []))
    mappables = []
    drawn = [xyz[edges.edge_index[:, edges.ids].reshape(-1)], xyz[node_ids]]

    def line_collection(segments, **kwargs):
        segments = segments[..., dims]
        if is_3d:
            ax.add_collection3d(Line3DCollection(segments, **kwargs), autolim=False)
        else:
            ax.add_collection(LineCollection(segments, **kwargs), autolim=False)

    def text(points, labels, color, **kwargs):
        for point, label in zip(points, labels):
            if label:
                ax.text(*point[dims], label, color=color, fontsize=fontsize, zorder=8, clip_on=False, **kwargs)

    # reference geometry
    reference = _ghost(data, ghost)
    if reference is not None:
        ghost_xyz, ghost_edges = reference
        line_collection(
            np.stack([ghost_xyz[ghost_edges[0]], ghost_xyz[ghost_edges[1]]], axis=1),
            colors=_GHOST_COLOR, linestyles="--", linewidths=0.8, zorder=1,
        )
        drawn.append(ghost_xyz)
        legend_entries.append((Line2D([], [], color=_GHOST_COLOR, linestyle="--", lw=0.8), ghost_label))

    # edges
    if len(edges.ids):
        force = getattr(data, "force", None)
        has_force = force is not None and isinstance(force, torch.Tensor) and force.dtype != torch.bool
        if edge_color is None and has_force:
            values = edges.values(force).astype(np.float64)
            colors = np.array([
                to_rgba(TENSION_COLOR if f > 0 else COMPRESSION_COLOR if f < 0 else _ZERO_COLOR) for f in values
            ]).reshape(-1, 4)
            for color, label, present in (
                (TENSION_COLOR, "Tension", values > 0),
                (COMPRESSION_COLOR, "Compression", values < 0),
                (_ZERO_COLOR, "Zero force", values == 0),
            ):
                if present.any():
                    legend_entries.append((Line2D([], [], color=color, lw=2), label))
        else:
            constant, values, name = _color_values(data, edge_color, edges.values, _EDGE_COLOR)
            if constant is not None:
                colors = constant
            else:
                colors, entries, mappable = _map_colors(values, name, cmap, clim, _EDGE_COLOR)
                legend_entries += [(Line2D([], [], color=c, lw=2), label) for c, label in entries]
                mappables.append(mappable)

        if edge_width is None:
            widths = _widths(edges.values(force), lw_scale) if has_force else 1.5 * lw_scale
        elif isinstance(edge_width, (int, float)):
            widths = edge_width * lw_scale
        else:
            widths = _widths(_lookup(data, edge_width, edges.values)[0], lw_scale)

        src, dst = edges.edge_index[:, edges.ids]
        line_collection(np.stack([xyz[src], xyz[dst]], axis=1), colors=colors, linewidths=widths, zorder=2)

        labels = _labels(data, edge_labels, edges.values, label_format, {"index": edges.ids})
        if labels is not None:
            text((xyz[src] + xyz[dst]) / 2, labels, "#2b2a28", ha="center", va="center",
                 bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 0.4})

    # nodes
    def node_values(value):
        value = _numpy(value)
        if len(value) != num_nodes:
            raise ValueError(f"A node value has {len(value)} entries, expected one per node ({num_nodes}).")
        value = value.reshape(num_nodes, -1)
        return (value[:, 0] if value.shape[1] == 1 else value)[node_ids]

    # the nodes are what a pattern places, so they are drawn with its elements
    if (node_color is not None or node_size is not None or pattern) and len(node_ids):
        constant, values, name = _color_values(data, node_color, node_values, _NODE_COLOR)
        if constant is not None:
            colors = constant
        else:
            colors, entries, mappable = _map_colors(values, name, cmap, clim, _NODE_COLOR)
            legend_entries += [
                (Line2D([], [], color=c, marker="o", linestyle="none"), label) for c, label in entries
            ]
            mappables.append(mappable)

        if node_size is None:
            sizes = 12
        elif isinstance(node_size, (int, float)):
            sizes = node_size
        else:
            sizes = _widths(_lookup(data, node_size, node_values)[0], 1.0, 4.0, 80.0)

        points = xyz[node_ids][:, dims].T
        kwargs = {"depthshade": False} if is_3d else {}
        ax.scatter(*points, s=sizes, c=colors, zorder=3, **kwargs)

    labels = _labels(data, node_labels, node_values, label_format, {
        "index": node_ids,
        "name": _node_names(data, node_ids) if node_labels == "name" else None,
    })
    if labels is not None:
        text(xyz[node_ids], labels, "#1f1e1c", ha="left", va="bottom")

    # supports
    support = _supports(data, supports)
    if support is not None:
        is_support, fixed = support
        kwargs = {"depthshade": False} if is_3d else {}
        for selection, marker, label in ((is_support & ~fixed, "o", "Support"), (is_support & fixed, "s", "Fixed support")):
            selection = selection & _mask(node_mask, num_nodes, "node")
            if selection.any():
                ax.scatter(*xyz[selection][:, dims].T, s=45, marker=marker, color="black", zorder=6, **kwargs)
                legend_entries.append((Line2D([], [], color="black", marker=marker, linestyle="none"), label))

    # node vectors
    vector_sets = _vector_sets(data, loads, reactions, vectors, num_nodes, node_ids)
    if vector_sets:
        base = xyz[node_ids]
        inward = _inward_directions(xyz, edges.edge_index)[node_ids]
        # one scale for all arrows, so that their lengths compare as force magnitudes
        magnitudes = [np.nan_to_num(np.linalg.norm(vector, axis=1)) for _, vector, _, _ in vector_sets]
        largest = max((m.max() for m in magnitudes if m.size), default=0.0)
        scale = vector_scale if vector_scale is not None else (0.15 * extent / largest if largest > 0 else 0.0)
        for (label, vector, color, outside), magnitude in zip(vector_sets, magnitudes):
            keep = magnitude > 1e-12 * largest
            if largest == 0 or not keep.any():
                continue
            start, delta = base[keep], scale * vector[keep]
            if outside:
                # keep reactions outside the structure: one pushing on it ends at its support
                pushes = np.einsum("ij,ij->i", vector[keep], inward[keep]) > 0
                start = np.where(pushes[:, None], start - delta, start)
            if is_3d:
                ax.quiver(*start.T, *delta.T, color=color, arrow_length_ratio=0.2, linewidths=1.2, zorder=7)
            else:
                ax.quiver(*start[:, dims].T, *delta[:, dims].T, color=color, angles="xy", scale_units="xy",
                          scale=1, width=0.003, zorder=7)
            drawn += [start, start + delta]
            legend_entries.append((Line2D([], [], color=color, lw=1.5, marker=">", markersize=5), label))

    # pattern elements
    if pattern:
        from torch_structure.plot.pattern import draw_pattern_elements

        if isinstance(pattern, dict):
            options = dict(pattern)
        else:
            options = {"patterns": None if pattern is True else pattern}
        options.setdefault("lw_scale", lw_scale)
        handles, labels, points = draw_pattern_elements(ax, data, xyz, extent, **options)
        legend_entries += list(zip(handles, labels))
        drawn += points

    # axes, colorbars, legend
    if not show_axes:
        ax.set_axis_off()
    elif not is_3d:
        ax.set_xlabel("xyz"[dims[0]])
        ax.set_ylabel("xyz"[dims[1]])
    else:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    _set_limits(ax, drawn, dims, equal_axes)

    if colorbar:
        for mappable in mappables:
            if mappable is not None:
                bar = ax.figure.colorbar(mappable, ax=ax, shrink=0.6, pad=0.02)
                bar.set_label(mappable._ts_label)

    ax._ts_legend = legend_entries
    if legend and legend_entries:
        unique = dict((label, handle) for handle, label in legend_entries)
        ax.legend(unique.values(), unique.keys(), frameon=False, loc="upper left", fontsize=9)

    if title is not None:
        ax.set_title(title)

    if path is not None:
        path = str(path)
        ax.figure.savefig(path if os.path.splitext(path)[1] else f"{path}.png", bbox_inches="tight")

    if show:
        plt.show()

    return ax
