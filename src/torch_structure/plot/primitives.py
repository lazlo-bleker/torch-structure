"""Low-level drawing primitives.

Each function takes a matplotlib ``ax``, already-packaged geometry (plain numpy
arrays in plot space) and style properties, and issues a *single* batched
matplotlib call. They do no tensor work and no data wrangling.

The five object types are Node, Edge, Face, Arrow (vector) and Text. Each
function handles both 2D and 3D axes transparently, branching internally on
the trailing coordinate dimension of the geometry it's given.
"""

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


def draw_node(ax, points, *, marker="o", size=6, color="black", zorder=None):
    """
    Scatter marker glyphs at a set of points, in a single call.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on. A 3D ``Axes3D`` and a
            plain 2D ``Axes`` are both supported.
        points (numpy.ndarray): Array of shape (K, 2) or (K, 3) with the
            marker positions, in plot space.
        marker (str, optional): Matplotlib marker style. Default is "o".
        size (float, optional): Marker size in points. Default is 6.
        color (str, optional): Marker color. Default is "black".
        zorder (float, optional): Draw order passed to the artist. Default is None.

    Returns:
        matplotlib.lines.Line2D: The marker artist, or None if `points` is empty.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return None
    (artist,) = ax.plot(
        *points.T,
        marker=marker,
        markersize=size,
        color=color,
        linestyle="None",
        zorder=zorder,
        clip_on=False,
    )
    return artist


def draw_edge(ax, segments, *, colors=None, linewidths=None, linestyles=None, zorder=None):
    """
    Draw every edge as one batched line collection.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on. Uses a ``Line3DCollection``
            for 3D axes and a ``LineCollection`` for 2D axes, selected from the
            shape of `segments`.
        segments (numpy.ndarray): Array of shape (E, 2, 2) or (E, 2, 3) with a
            [start, end] coordinate pair per edge.
        colors (optional): Single color, or length-E sequence of colors, one
            per edge. Default is None (matplotlib's default color cycle).
        linewidths (optional): Single width, or length-E sequence of widths,
            one per edge. Default is None.
        linestyles (optional): Single style, or length-E sequence of styles
            (e.g. "-", ":"), one per edge. Default is None (solid).
        zorder (float, optional): Draw order passed to the collection. Default is None.

    Returns:
        matplotlib.collections.Collection: The ``Line3DCollection`` or
        ``LineCollection`` that was added to `ax`.
    """
    segments = np.asarray(segments, dtype=float)
    three_d = segments.shape[-1] == 3
    collection_cls = Line3DCollection if three_d else LineCollection
    style = {"colors": colors, "linewidths": linewidths}
    if linestyles is not None:
        style["linestyles"] = linestyles
    collection = collection_cls(segments, clip_on=False, **style)
    if zorder is not None:
        collection.set_zorder(zorder)
    if three_d:
        ax.add_collection3d(collection)
    else:
        ax.add_collection(collection)
    return collection


def draw_face(ax, polygons, *, facecolor="grey", edgecolor="k", linewidth=0.5, alpha=1.0):
    """
    Draw a set of filled polygons as one batched collection.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on. Uses a ``Poly3DCollection``
            for 3D axes and a ``PolyCollection`` for 2D axes, selected from the
            shape of the first polygon.
        polygons (Sequence[numpy.ndarray]): Sequence of vertex arrays, each of
            shape (V, 2) or (V, 3).
        facecolor (optional): Fill color. Default is "grey".
        edgecolor (optional): Outline color. Default is "k".
        linewidth (float, optional): Outline width. Default is 0.5.
        alpha (float, optional): Fill opacity. Default is 1.0.

    Returns:
        matplotlib.collections.Collection: The ``Poly3DCollection`` or
        ``PolyCollection`` that was added to `ax`, or None if `polygons` is empty.
    """
    polygons = [np.asarray(p, dtype=float) for p in polygons]
    if not polygons:
        return None
    three_d = polygons[0].shape[-1] == 3
    collection_cls = Poly3DCollection if three_d else PolyCollection
    collection = collection_cls(
        polygons,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
        clip_on=False,
    )
    if three_d:
        ax.add_collection3d(collection)
    else:
        ax.add_collection(collection)
    return collection


def draw_text(ax, points, texts, *, color="black", fontsize=8):
    """
    Place one text label at each of a set of points.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on. A 3D ``Axes3D`` and a
            plain 2D ``Axes`` are both supported.
        points (numpy.ndarray): Array of shape (K, 2) or (K, 3) with one
            label position per entry of `texts`.
        texts (Sequence[str]): Label string for each point, same length as `points`.
        color (str, optional): Text color. Default is "black".
        fontsize (float, optional): Font size in points. Default is 8.
    """
    for point, text in zip(np.asarray(points, dtype=float), texts):
        ax.text(*point, text, color=color, fontsize=fontsize, clip_on=False)


def draw_arrow(ax, origins, vectors, *, color="black", length=1.0, arrow_length_ratio=0.5,
              normalize=False, linewidth=None):
    """
    Draw a set of vector arrows with a single ``quiver`` call.

    Uses a 3D ``quiver`` (``length`` / `arrow_length_ratio` / `normalize`
    control the arrowheads) when `origins` is (K, 3), and a 2D ``quiver`` drawn
    at data scale (``angles="xy", scale_units="xy", scale=1`` — `length`,
    `arrow_length_ratio` and `normalize` do not apply) when `origins` is (K, 2).

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on. A 3D ``Axes3D`` and a
            plain 2D ``Axes`` are both supported, selected from the shape of
            `origins`.
        origins (numpy.ndarray): Array of shape (K, 2) or (K, 3) with the tail
            position of each arrow.
        vectors (numpy.ndarray): Array of shape (K, 2) or (K, 3), matching
            `origins`, with the direction and magnitude of each arrow.
        color (str, optional): Arrow color. Default is "black".
        length (float, optional): 3D only — uniform scale factor applied to
            every arrow. Default is 1.0.
        arrow_length_ratio (float, optional): 3D only — fraction of each
            arrow's length given to the arrowhead. Default is 0.5.
        normalize (bool, optional): 3D only — if True, every arrow is
            rescaled to unit length before `length` is applied. Default is False.
        linewidth (float, optional): Shaft/outline stroke width. Default is
            None (matplotlib's own default).

    Returns:
        matplotlib.quiver.Quiver: The quiver artist, or None if `origins` is empty.
    """
    origins = np.asarray(origins, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    if origins.size == 0:
        return None
    if origins.shape[-1] == 3:
        return ax.quiver(
            origins[:, 0], origins[:, 1], origins[:, 2],
            vectors[:, 0], vectors[:, 1], vectors[:, 2],
            color=color,
            length=length,
            arrow_length_ratio=arrow_length_ratio,
            normalize=normalize,
            linewidth=linewidth,
            clip_on=False,
        )
    return ax.quiver(
        origins[:, 0], origins[:, 1],
        vectors[:, 0], vectors[:, 1],
        color=color,
        angles="xy",
        scale_units="xy",
        linewidth=linewidth,
        scale=1,
        clip_on=False,
    )
