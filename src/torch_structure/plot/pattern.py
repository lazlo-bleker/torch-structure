"""
Plots a structure together with the pattern elements of its registered patterns: the
mirror planes, rotation axes and translations, read level by level from the pattern
matrices.
"""

import math

import numpy as np
import torch
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


# the package's orange, the same for every kind: the kinds are told apart by how they are drawn
PATTERN_COLOR = "#E7461E"

_NODE_COLOR = "#3d3c39"
_LABEL_COLOR = "#52514e"

_PLANE_KINDS = ("mirror", "rotoreflection")
_AXIS_KINDS = ("rotation", "screw", "rotoreflection")


def _shown_kind(kind):
    # a glide is drawn as the translation it slides by, with its mirror left out
    return "translation" if kind == "glide" else kind


def _kind_name(rotation, mirror, translation, inversion=False):
    if mirror:
        if rotation:
            return "inversion" if inversion else "rotoreflection"
        return "glide" if translation else "mirror"
    if rotation:
        return "screw" if translation else "rotation"
    return "translation" if translation else "identity"


def classify_affine(matrices, atol=1e-5):
    """
    Splits affine matrices into their rotation, mirror and translation parts.

    Every isometry of space is the identity, a translation, a rotation about an axis, a screw
    (a rotation plus a slide along its axis), a mirror, a glide (a mirror plus a slide within
    its plane), a rotoreflection (a rotation plus the mirror in the plane normal to its axis)
    or an inversion through a point. The linear part ``R`` tells most of it: ``det(R) = -1``
    means it mirrors, and the angle of ``R`` (or of ``-R``) whether it rotates. Whether it
    translates cannot be read from the translation column ``t`` alone, i.e. from where the
    origin goes, since a rotation about an axis, or a mirror in a plane, that misses the origin
    moves the origin too. Only the part of ``t`` along the directions ``R`` keeps fixed is a
    slide: no choice of the axis or plane can absorb it. The rest of ``t`` only places the axis
    or plane.

    Args:
        matrices (torch.Tensor): affine matrices of shape [K, 4, 4].
        atol (float): tolerance below which an angle (in radians), or a slide relative to the
            length of the translation column, counts as zero. Defaults to ``1e-5``.

    Returns:
        dict: per matrix
            - ``"kind"`` (list[str]): ``"identity"``, ``"translation"``, ``"rotation"``,
              ``"screw"``, ``"mirror"``, ``"glide"``, ``"rotoreflection"``, ``"inversion"``,
              or ``"affine"`` for a matrix that is no isometry.
            - ``"rotation"``, ``"mirror"``, ``"translation"`` (torch.Tensor): bool, shape [K],
              which parts the matrix is made of.
            - ``"isometry"`` (torch.Tensor): bool, shape [K].
            - ``"axis"`` (torch.Tensor): shape [K, 3], the unit rotation axis, or the normal
              of the mirror plane.
            - ``"angle"`` (torch.Tensor): shape [K], the rotation angle about ``axis`` in
              radians, between 0 and pi.
            - ``"point"`` (torch.Tensor): shape [K, 3], the point of the axis or plane
              closest to the origin, or the point a rotoreflection or inversion keeps fixed.
            - ``"slide"`` (torch.Tensor): shape [K, 3], the translation part.
    """

    A = torch.as_tensor(matrices).detach().cpu().to(torch.float64).reshape(-1, 4, 4)
    R, t = A[:, :3, :3], A[:, :3, 3]
    eye = torch.eye(3, dtype=A.dtype)

    isometry = (R.mT @ R - eye).abs().amax(dim=(1, 2)) <= 10 * atol
    mirror = torch.linalg.det(R) < 0

    # a mirroring R is -1 times a rotation: the rotation by pi more about the mirror's normal
    proper = torch.where(mirror.view(-1, 1, 1), -R, R)
    cos = ((proper.diagonal(dim1=1, dim2=2).sum(dim=1) - 1) / 2).clamp(-1.0, 1.0)
    # the axis is the direction the rotation keeps: the singular vector of proper - I to its smallest singular value
    axis = torch.linalg.svd(proper - eye).Vh[:, 2]
    skew = (proper - proper.mT) / 2
    sin = (axis * torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=1)).sum(dim=1)
    angle = torch.atan2(sin, cos)
    angle = torch.where(mirror, torch.remainder(angle, 2 * math.pi) - math.pi, angle)

    # the same rotation, turned the other way about the opposite axis
    axis = torch.where((angle < 0).unsqueeze(1), -axis, axis)
    angle = angle.abs()
    rotation = angle > atol

    # without a sense of turning, the axis points to where its largest component is positive
    unturned = ~rotation | (angle > math.pi - atol)
    largest = axis.gather(1, axis.abs().argmax(dim=1, keepdim=True))
    axis = torch.where(unturned.unsqueeze(1) & (largest < 0), -axis, axis)

    # the slide is the part of t along the directions R keeps: the axis of a rotation, the plane of a mirror,
    # every direction for no rotation at all, none for a rotoreflection
    along = (t * axis).sum(dim=1, keepdim=True) * axis
    slide = torch.where(
        rotation.unsqueeze(1),
        torch.where(mirror.unsqueeze(1), torch.zeros_like(t), along),
        torch.where(mirror.unsqueeze(1), t - along, t),
    )
    translation = slide.norm(dim=1) > atol * t.norm(dim=1).clamp(min=1.0)
    slide = torch.where(translation.unsqueeze(1), slide, torch.zeros_like(slide))

    # the rest of t places the axis or plane: the solution of (I - R) x = t - slide closest to the origin
    point = (torch.linalg.pinv(eye - R, atol=atol) @ (t - slide).unsqueeze(2)).squeeze(2)

    inversion = mirror & (angle > math.pi - atol)
    kind = [
        _kind_name(r, m, s, i) if iso else "affine"
        for r, m, s, i, iso in zip(
            rotation.tolist(), mirror.tolist(), translation.tolist(), inversion.tolist(), isometry.tolist()
        )
    ]

    return {
        "kind": kind,
        "rotation": rotation & isometry,
        "mirror": mirror & isometry,
        "translation": translation & isometry,
        "isometry": isometry,
        "axis": axis,
        "angle": angle,
        "point": point,
        "slide": slide,
    }


def _pattern_id(data, pattern):
    name_to_pattern = data.metadata.get("name_to_pattern", {})
    if pattern not in name_to_pattern:
        raise ValueError(f"Pattern '{pattern}' is not registered.")
    return name_to_pattern[pattern]


def pattern_levels(data, pattern, atol=1e-5):
    """
    Infers for every nesting level of a registered pattern what kind of operation it is.

    A level is one factor of the nested pattern, e.g. one ``create_*`` call inside
    ``nest_patterns``. Its own copies sit at the expansion positions that are 0 on every other
    level, so a level is read from just its ``size`` matrices, and the whole pattern from the
    sum of its level sizes instead of their product. The steps between consecutive copies of a
    level are classified with ``classify_affine``. Each level is read once, in the frame of the
    first copy of the levels around it: that the outer levels copy it elsewhere changes where
    its axis or plane is, not what it is.

    Args:
        data (StructData): a graph with registered patterns.
        pattern (str): the name of the pattern.
        atol (float): see ``classify_affine``.

    Returns:
        list[dict]: one entry per level, innermost first, with
            - ``"level"``, ``"size"``, ``"closes"``: the level's index, number of copies, and
              whether it closes back onto its first copy.
            - ``"place"``: the step in expansion position between two consecutive copies of the
              level.
            - ``"kind"``: the kind of operation (see ``classify_affine``), and ``"rotation"``,
              ``"mirror"``, ``"translation"``: the parts the level is made of.
            - ``"steps"``: the ``classify_affine`` output for the steps from copy
              ``step_from[i]`` to copy ``step_to[i]`` of the level: from every copy to the next,
              and for a closing level of more than two copies from the last back to the first.
            - ``"step_from"``, ``"step_to"`` (list[int]).

    Raises:
        ValueError: if ``pattern`` is not registered.
    """

    pattern_id = _pattern_id(data, pattern)
    matrices = data.pattern_matrices[data.pattern_matrix_id == pattern_id].detach().cpu().to(torch.float64)
    level_sizes, level_closes = data._infer_level_sizes(pattern_id)

    levels = []
    place = 1
    for level, (size, closes) in enumerate(zip(level_sizes, level_closes)):
        own = matrices[torch.arange(size) * place]

        step_from, step_to = list(range(size - 1)), list(range(1, size))
        if closes and size > 2:
            step_from.append(size - 1)
            step_to.append(0)

        steps = classify_affine(own[step_to] @ torch.linalg.inv(own[step_from]), atol)
        isometry = bool(steps["isometry"].all())
        rotation = bool(steps["rotation"].any())
        mirror = bool(steps["mirror"].any())
        translation = bool(steps["translation"].any())
        inversion = len(steps["kind"]) > 0 and all(kind == "inversion" for kind in steps["kind"])

        levels.append({
            "level": level,
            "size": size,
            "closes": closes,
            "place": place,
            "kind": _kind_name(rotation, mirror, translation, inversion) if isometry else "affine",
            "rotation": rotation,
            "mirror": mirror,
            "translation": translation,
            "steps": steps,
            "step_from": step_from,
            "step_to": step_to,
        })
        place *= size

    return levels


class _Arrow3D(FancyArrowPatch):
    # an arrow between two points in 3D, projected whenever the view changes

    def __init__(self, start, end, **kwargs):
        super().__init__((0, 0), (0, 0), **kwargs)
        self._start, self._end = start, end

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = proj3d.proj_transform(*np.stack([self._start, self._end], axis=1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min(zs))


def _expansion_table(data, pattern_id, matrices, coords):
    # the coordinates of every expansion of the pattern at every expansion position, shape [expansions, pattern_size, 3]; a
    # copy that was merged into its twin takes its twin's node
    pattern_size = matrices.shape[0]
    tracked = data.pattern_id.view(-1) == pattern_id
    expansions = torch.unique(data.expansion_id.view(-1)[tracked])
    if expansions.numel() == 0:
        # no nodes yet: show what the pattern does to the origin
        return matrices[:, :3, 3].unsqueeze(0).numpy()

    lookup, rows = data._build_expansion_lookup(expansions)
    lookup = lookup[rows, :pattern_size].numpy()
    table = coords[np.clip(lookup, 0, None)]
    table[lookup < 0] = np.nan
    return table


def _nan_mean(values, axis):
    finite = np.isfinite(values).all(axis=-1, keepdims=True)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    count = finite.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _canonical(direction):
    return direction if direction[np.argmax(np.abs(direction))] >= 0 else -direction


def _key(values, quantum):
    return tuple(np.round(np.asarray(values, dtype=float).ravel() / quantum).astype(np.int64).tolist())


def _perpendicular(u):
    helper = np.eye(3)[np.argmin(np.abs(u))]
    e1 = np.cross(u, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


def _step_path(kind, point, axis, angle, slide, start, end):
    # where `start` travels under the step, with the motion unrolled from none at tau = 0 to the full step at
    # tau = 1; trimmed at both ends so that the steps of a level stay apart
    rotational = kind in _AXIS_KINDS or kind == "inversion"
    num = max(8, int(math.ceil(abs(angle) / (2 * math.pi) * 96))) if rotational else 2
    tau = np.linspace(0.06, 0.94, num)[:, None]

    if kind == "affine":
        return start + tau * (end - start)

    offset = start - point
    height = offset @ axis
    radial = offset - height * axis

    if kind == "inversion":
        return point + (1 - 2 * tau) * offset
    if kind == "translation":
        return start + tau * slide
    if kind == "glide":
        # within the plane, from the foot of start
        return point + radial + tau * slide

    turned = radial * np.cos(tau * angle) + np.cross(axis, radial) * np.sin(tau * angle)
    # a rotoreflection carries start through the plane to the other side
    height = height * (1 - 2 * tau) if kind == "rotoreflection" else height
    return point + turned + height * axis + tau * slide


def _full_step(kind, point, axis, angle, slide, start):
    # the image of start under the whole step of a rotation, screw or rotoreflection
    offset = start - point
    height = offset @ axis
    radial = offset - height * axis
    turned = radial * np.cos(angle) + np.cross(axis, radial) * np.sin(angle)
    height = -height if kind == "rotoreflection" else height
    return point + turned + height * axis + slide


def _level_elements(info, matrices, table, copies, scale):
    # the planes, axes and paths of one level, in each copy that the levels around it make of it
    size, inner = info["size"], info["place"]
    num_outer = matrices.shape[0] // (size * inner)

    cells = table.reshape(table.shape[0], num_outer, size, inner, 3)
    # the nodes each outer copy of the level acts on
    domains = cells.transpose(1, 0, 2, 3, 4).reshape(num_outer, -1, 3)

    # the centroid of each copy of the level, in each outer copy, of the expansions the level moves: a node on the
    # axis or plane, which every copy shares, would only pull the centroids onto it
    moved = ~np.all(np.abs(cells - cells[:, :, :1]) <= 1e-6 * scale, axis=(2, 3, 4))
    moved |= ~moved.any(axis=0, keepdims=True)
    anchors = _nan_mean(np.where(moved[:, :, None, None, None], cells, np.nan), axis=(0, 3))

    outer = matrices[np.arange(num_outer) * size * inner]
    if not copies:
        outer = outer[:1]

    steps = info["steps"]
    rot, shift = outer[:, :3, :3], outer[:, :3, 3]
    # conjugating a step by an outer copy moves its axis or plane and, for a mirroring copy, turns it the other way
    point = np.einsum("qij,sj->qsi", rot, steps["point"].numpy()) + shift[:, None]
    axis = np.einsum("qij,sj->qsi", rot, steps["axis"].numpy())
    slide = np.einsum("qij,sj->qsi", rot, steps["slide"].numpy())
    angle = np.linalg.det(rot)[:, None] * steps["angle"].numpy()[None]

    planes, axes, paths, centres = {}, {}, {}, []
    for q in range(outer.shape[0]):
        domain = domains[q][np.isfinite(domains[q]).all(axis=1)]
        carried = None

        for s, kind in enumerate(steps["kind"]):
            p, u, a, v = point[q, s], axis[q, s], angle[q, s], slide[q, s]

            if kind in _PLANE_KINDS:
                n = _canonical(u)
                key = _key(n, 1e-3) + _key([n @ p], 1e-3 * scale)
                planes.setdefault(key, {"point": p, "normal": n, "kind": kind, "domains": []})["domains"].append(domain)

            if kind in _AXIS_KINDS:
                d = _canonical(u)
                key = _key(d, 1e-3) + _key(p - (p @ d) * d, 1e-3 * scale)
                axes.setdefault(key, {"point": p, "direction": d, "kind": kind, "domains": []})["domains"].append(domain)

            if kind == "inversion":
                centres.append(p)

            d_from, d_to = info["step_from"][s], info["step_to"][s]
            start, end = anchors[q, d_from], anchors[q, d_to]
            if kind == "mirror" or not np.isfinite(start).all():
                continue
            # the step from the last copy back to the first is only drawn where it closes a turn
            if d_to < d_from and kind not in _AXIS_KINDS:
                continue

            if kind in _AXIS_KINDS:
                offset = start - p
                if np.linalg.norm(offset - (offset @ u) * u) < 1e-3 * scale:
                    # the copies sit on the axis: turn a point just outside the nodes instead, carried on from step
                    # to step
                    if carried is None:
                        around = (domain - p) - np.outer((domain - p) @ u, u)
                        radius = 1.2 * np.linalg.norm(around, axis=1).max(initial=0.0) or 0.08 * scale
                        carried = start + radius * _perpendicular(u)[0]
                    start = carried

            path = _step_path(kind, p, u, a, v, start, end)
            if carried is not None:
                carried = _full_step(kind, p, u, a, v, start)

            key = _key(path[[0, len(path) // 2, -1]], 1e-3 * scale)
            paths.setdefault(key, {"xyz": path, "kind": _shown_kind(kind)})

    return list(planes.values()), list(axes.values()), list(paths.values()), centres


def _plane_corners(plane, scale):
    points = np.concatenate(plane["domains"], axis=0)
    p, n = plane["point"], plane["normal"]
    e1, e2 = _perpendicular(n)

    centre = points.mean(axis=0) if len(points) else p
    centre = centre - ((centre - p) @ n) * n
    uv = (points - centre) @ np.stack([e1, e2], axis=1) if len(points) else np.zeros((1, 2))

    low, high = uv.min(axis=0), uv.max(axis=0)
    margin = 0.08 * (high - low) + 0.04 * scale
    mid = (low + high) / 2
    half = np.maximum((high - low) / 2 + margin, 0.1 * scale)
    (u0, v0), (u1, v1) = mid - half, mid + half

    return np.stack([centre + a * e1 + b * e2 for a, b in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))])


def _axis_segment(axis, scale):
    points = np.concatenate(axis["domains"], axis=0)
    p, d = axis["point"], axis["direction"]
    along = (points - p) @ d if len(points) else np.zeros(1)

    low, high = along.min(), along.max()
    margin = 0.15 * (high - low) + 0.05 * scale
    mid = (low + high) / 2
    half = max((high - low) / 2 + margin, 0.1 * scale)

    return np.stack([p + (mid - half) * d, p + (mid + half) * d])


def _number(value):
    return f"{value:.3g}" if abs(value) >= 5e-4 else "0"


def _vector(vector):
    return "(" + ", ".join(_number(float(v)) for v in vector) + ")"


def _coordinate_axis(direction):
    # the index of the coordinate axis a direction runs along, or None
    i = int(np.argmax(np.abs(direction)))
    return i if abs(abs(direction[i]) - 1) < 1e-6 else None


def _plane_text(point, normal):
    i = _coordinate_axis(normal)
    if i is not None:
        # the point lies on the plane, so its coordinate along the normal places the plane
        return f"plane {'xyz'[i]} = {_number(float(point[i]))}"
    return f"plane through {_vector(point)} normal to {_vector(normal)}"


def _axis_text(point, direction):
    i = _coordinate_axis(direction)
    text = f"about {'xyz'[i]}" if i is not None else f"about {_vector(direction)}"
    return text + (f" through {_vector(point)}" if np.linalg.norm(point) >= 5e-4 else "")


def _level_legend(info, prefix):
    # the level as it was read from its matrices, in the frame of the first copy of the levels around it
    kind, steps = _shown_kind(info["kind"]), info["steps"]
    point, axis = steps["point"][0].numpy(), steps["axis"][0].numpy()
    angle, slide = math.degrees(float(steps["angle"][0])), steps["slide"][0].numpy()

    details = [] if kind in ("mirror", "inversion") else [f"{info['size']} positions"]
    if kind == "translation":
        details.append(f"step {_vector(slide)}")
    elif kind == "rotation":
        details.append(f"{_number(angle)}° {_axis_text(point, axis)}")
    elif kind == "screw":
        details.append(f"{_number(angle)}° {_axis_text(point, axis)}, rising {_number(float(slide @ axis))}")
    elif kind == "mirror":
        details.append(_plane_text(point, axis))
    elif kind == "rotoreflection":
        details.append(f"{_number(angle)}° {_axis_text(point, axis)}, {_plane_text(point, axis)}")
    elif kind == "inversion":
        details.append(f"through {_vector(point)}")

    # only the closing behaviour that differs from what the kind does by default
    if info["size"] > 2 and info["rotation"] and not info["closes"]:
        details.append("open")
    if info["size"] > 2 and not info["rotation"] and info["translation"] and info["closes"]:
        details.append("closing")

    return f"{prefix}T{info['level']} {kind}: " + ", ".join(details)


def _legend_handle(kind, color):
    if kind == "mirror":
        return Patch(facecolor=to_rgba(color, 0.25), edgecolor=color)
    if kind in _AXIS_KINDS:
        return Line2D([], [], color=color, linestyle=":", lw=2.0)
    if kind == "inversion":
        return Line2D([], [], color=color, marker="o", markerfacecolor="white", linestyle="none")
    return Line2D([], [], color=color, lw=1.8, marker=">", markersize=5)


def draw_pattern_elements(
    ax,
    data,
    coords,
    scale,
    patterns=None,
    levels=None,
    copies=False,
    show_labels=True,
    show_seeds=False,
    color=PATTERN_COLOR,
    plane_alpha=0.12,
    lw_scale=1.0,
    node_size=10,
    atol=1e-5,
):
    """
    Draws the pattern elements of registered patterns on an existing 3D axis, level by level:
    mirror planes as translucent planes, rotation axes as dotted lines with arcs that follow one
    step of the rotation, and translations as arrows from each copy to the next. All elements
    are drawn in one color, ``PATTERN_COLOR``; a hybrid level, e.g. a screw (rotation +
    translation) or a rotoreflection (rotation + mirror), combines the elements of its parts. A
    glide (mirror + translation) is drawn as the translation it slides by: arrows only, without
    its plane.

    The kind of each level is inferred from the pattern matrices with ``pattern_levels``,
    which reads only the level's own matrices. Each level is drawn once, in the first copy of
    the levels around it, where the legend reads it too: a mirror nested inside a translation
    gets one mirror plane, not one in every translated copy, see ``copies``. Each element is
    sized to the nodes it acts on, and an arrow or arc runs between the centroids of the
    copies it maps onto each other.

    Usually called through [plot_structure][torch_structure.plot.structure.plot_structure]
    with its ``pattern`` argument, e.g. ``data.plot(pattern=True)``.

    Args:
        ax (mpl_toolkits.mplot3d.Axes3D): the axis to draw on.
        data (StructData): the structure, with its patterns registered via ``add_patterns``.
        coords (numpy.ndarray): node coordinates of shape [num_nodes, 3].
        scale (float): the size of the structure, which the size of elements around nodes
            that all lie on one axis or plane falls back to.
        patterns (str | list[str], optional): the patterns to draw. Defaults to all
            registered ones.
        levels (list[int] | dict[str, list[int]], optional): the levels to draw, the same for
            every pattern, or per pattern name, where a pattern left out of the dict draws all
            of its levels. Defaults to all levels.
        copies (bool): if True, draws the elements of a level in every copy that the levels
            around it make of it; if False, only in the first one. Default is False.
        show_labels (bool): if True, labels the first copy of each level's element with the
            level, e.g. ``T1``. Default is True.
        show_seeds (bool): if True, rings the nodes at expansion position 0, i.e. the nodes every
            other node of their expansion is a copy of. Default is False.
        color (str): color of all pattern elements. Default is ``PATTERN_COLOR``, the
            package's orange.
        plane_alpha (float): opacity of the mirror planes. Default is 0.12.
        lw_scale (float): scale factor for all line widths. Default is 1.0.
        node_size (float): marker size the seed rings are sized from. Default is 10.
        atol (float): see ``classify_affine``.

    Returns:
        tuple[list, list[str], list[numpy.ndarray]]: the legend handles and labels, and the
        points drawn, for the axis limits.

    Raises:
        ValueError: if a name in ``patterns`` is not a registered pattern.
    """

    registered = list(getattr(data, "metadata", {}).get("name_to_pattern", {}))
    if patterns is None:
        patterns = registered
    elif isinstance(patterns, str):
        patterns = [patterns]
    for name in patterns:
        if name not in registered:
            raise ValueError(f"Pattern '{name}' is not registered.")

    coords = np.asarray(coords, dtype=np.float64)
    drawn_points = []
    handles, labels = [], []

    seeds = coords[(data.expansion_position.view(-1) == 0).cpu().numpy()] if hasattr(data, "expansion_position") else []
    if show_seeds and len(seeds):
        handles.append(ax.scatter(
            *seeds.T, s=6 * node_size, facecolors="none", edgecolors=_NODE_COLOR,
            linewidths=1.0, depthshade=False, zorder=3,
        ))
        labels.append("seed (expansion position 0)")
    planes, axes, paths, centres, level_labels = [], [], [], [], []

    for name in patterns:
        pattern_id = _pattern_id(data, name)
        matrices = data.pattern_matrices[data.pattern_matrix_id == pattern_id].detach().cpu().to(torch.float64)
        table = _expansion_table(data, pattern_id, matrices, coords)
        wanted = levels.get(name) if isinstance(levels, dict) else levels
        prefix = f"{name} " if len(patterns) > 1 else ""

        for info in pattern_levels(data, name, atol):
            if info["kind"] == "identity" or (wanted is not None and info["level"] not in wanted):
                continue

            level_planes, level_axes, level_paths, level_centres = _level_elements(
                info, matrices.numpy(), table, copies, scale
            )
            for plane in level_planes:
                plane["corners"] = _plane_corners(plane, scale)
            for axis in level_axes:
                axis["segment"] = _axis_segment(axis, scale)

            planes += level_planes
            axes += level_axes
            paths += level_paths
            centres += level_centres

            # the label sits at the first copy of the level's element, at the lower end of an axis, clear of the legend
            if level_planes:
                spot = level_planes[0]["corners"][2]
            elif level_axes:
                spot = level_axes[0]["segment"][0]
            elif level_paths:
                spot = level_paths[0]["xyz"][len(level_paths[0]["xyz"]) // 2]
            else:
                spot = None
            if spot is not None:
                level_labels.append((spot, f"{prefix}T{info['level']}"))

            shown = _shown_kind(info["kind"])
            handles.append(_legend_handle(shown, color))
            labels.append(_level_legend(info, prefix))

    if planes:
        ax.add_collection3d(Poly3DCollection(
            [plane["corners"] for plane in planes],
            facecolors=to_rgba(color, plane_alpha),
            edgecolors=to_rgba(color, 0.6),
            linewidths=0.8 * lw_scale,
            zorder=1,
        ), autolim=False)
        drawn_points += [plane["corners"] for plane in planes]

    if axes:
        ax.add_collection3d(Line3DCollection(
            [axis["segment"] for axis in axes],
            colors=color,
            linestyles=":",
            linewidths=2.0 * lw_scale,
            zorder=4,
        ), autolim=False)
        drawn_points += [axis["segment"] for axis in axes]

    if paths:
        ax.add_collection3d(Line3DCollection(
            [path_["xyz"] for path_ in paths],
            colors=color,
            linestyles=["--" if path_["kind"] == "inversion" else "-" for path_ in paths],
            linewidths=1.8 * lw_scale,
            zorder=5,
        ), autolim=False)
        for path_ in paths:
            xyz = path_["xyz"]
            ax.add_artist(_Arrow3D(
                xyz[max(0, len(xyz) - 3)], xyz[-1], arrowstyle="-|>", mutation_scale=11 * lw_scale,
                color=color, lw=1.8 * lw_scale, shrinkA=0, shrinkB=0, zorder=5,
            ))
        drawn_points += [path_["xyz"] for path_ in paths]

    for centre in centres:
        ax.scatter(*centre, s=40, facecolors="white", edgecolors=color, linewidths=1.5, depthshade=False, zorder=6)

    if show_labels:
        for spot, text in level_labels:
            ax.text(
                *spot, text, color=_LABEL_COLOR, fontsize=9, zorder=7,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 0.6},
            )

    return handles, labels, drawn_points
