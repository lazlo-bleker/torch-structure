import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import proj3d
from PIL import Image, ImageChops
from torch_structure.data.data import StructData
from torch_structure.plot.pattern import _Arrow3D

DARK = os.environ.get("TS_DARK") == "1"
BG = "none" if DARK else "white"
ACCENT_HEX = "#F05F42" if DARK else "#E7461E"
NODE_HEX = "#9A9A9A" if DARK else "#555555"
# the edges added last, with more contrast to the page than the nodes
NEW_EDGE_HEX = "#C4C4C4" if DARK else "#333333"
# the seed of a step, i.e. the node or edge given to it, is drawn darker than
# the copies the pattern makes of it, and larger: its marker size, in points
# squared, and its line width, in points
SEED_HEX = "#FFFFFF" if DARK else "#000000"
SEED_NODE_SIZE = 70
SEED_LW = 1.6
# the page's background, to fade colors toward
PAGE_HEX = "#1E2129" if DARK else "white"
# the space between a node and the ends of the pattern's arrows, in points
GAP = 4
# how far the translation's arrows are lifted off the edges they run along, in
# y: twice as far as in the other figures, which are drawn at twice the scale
LIFT = 0.2
# how far outside the outermost ring the rotation's arcs are moved out to, onto
# a circle around the net
MARGIN = 0.5

def faded(hex_color, amount=0.5):
    # blend toward the page's background: lighter on the light page, dimmer
    # on the dark one
    rgb = (1 - amount) * np.array(mcolors.to_rgb(hex_color)) + amount * np.array(mcolors.to_rgb(PAGE_HEX))
    return mcolors.to_hex(rgb)

def is_edge(edges, u, v):
    # which of the edges, one per column, join the nodes u and v, either way
    return ((edges[0] == u) & (edges[1] == v)) | ((edges[0] == v) & (edges[1] == u))

def mark_seed(collection, seed, colors):
    # draws the seed among the edges (z-order 2) or nodes (z-order 3) of a
    # collection, where seed is True, in SEED_HEX and wider or larger, and the
    # others in colors, one per edge or node
    collection.set_color([SEED_HEX if s else c for s, c in zip(seed, colors)])
    if collection.get_zorder() == 2:
        collection.set_linewidth(np.where(seed, SEED_LW, collection.get_linewidth()[0]))
    else:
        # an array, which mplot3d reorders by depth, as it does the colors
        collection.set_sizes(np.where(seed, SEED_NODE_SIZE, collection.get_sizes()[0]))

node_attrs ={"coords": torch.empty((0, 3), dtype=torch.float)}
data = StructData(node_attrs=node_attrs)

translation = data.create_translational_pattern(4, direction=torch.tensor([1.0, 0.0, 0.0]))
rotation = data.create_rotational_pattern(6)
data.add_patterns({"polar_net": data.nest_patterns([translation, rotation])})

data.add_nodes_by_pattern(pattern="polar_net", coords=torch.tensor([[1.0, 0.0, 0.0]]))


def save_figure(name, num_old=0, before=None, seed_node=None, seed_edge=None):
    # seed_node, seed_edge: the node, and the edge as its two nodes, given to
    # the step the figure shows
    fig = plt.figure(figsize=(6, 6), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=90, azim=-90)  # plan / top view, x to the right and y up

    data.plot_pattern(ax=ax, color=ACCENT_HEX, node_size=40, lw_scale=0.7, legend=False, show_labels=False)

    # plot_pattern() draws the edges at z-order 2 and the nodes at 3, both in
    # hardcoded greys, and the steps of the pattern at 5. The edges come in the
    # order they were added, one per pair of nodes. The edges the last edge
    # added, after the first num_old, are drawn darker and the ones before them
    # lighter. After a move from the coordinates before, all edges are drawn in
    # the nodes' grey. The seed edge and node, if given, are drawn darker still
    # and larger
    edges = data.edge_index[:, data.directed_mask.view(-1)]
    if before is None:
        edge_colors = [faded(NODE_HEX)] * num_old + [NEW_EDGE_HEX] * (edges.shape[1] - num_old)
    else:
        edge_colors = [NODE_HEX] * edges.shape[1]
    for collection in ax.collections:
        if collection.get_zorder() == 2 and seed_edge is not None:
            mark_seed(collection, is_edge(edges, *seed_edge).tolist(), edge_colors)
        elif collection.get_zorder() == 2:
            collection.set_color(edge_colors)
        elif collection.get_zorder() == 3 and seed_node is not None:
            mark_seed(collection, (torch.arange(data.num_nodes) == seed_node).tolist(), [NODE_HEX] * data.num_nodes)
        elif collection.get_zorder() == 3:
            # one color for all nodes, which matplotlib draws as markers on
            # whole pixels
            collection.set_color(NODE_HEX)

    if DARK:
        # mplot3d's background panes and the axes' own background patch are
        # opaque, make them transparent for the page's dark background
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor((1, 1, 1, 0))
        ax.patch.set_alpha(0.0)
        ax.set_facecolor((1, 1, 1, 0))

    # plot_pattern() draws the rotation's arcs between the centres of the rows,
    # across the net, so move them out onto a circle around it. It also trims
    # each step by a share of its length, which leaves short steps touching
    # their nodes, so cut both ends of every step back to GAP points clear of
    # the nodes, on the screen, lift the translation's steps off the edges they
    # run along, and move each arrowhead along. The arcs now run between the
    # points where the circle crosses the rows, so these count as ends too. The
    # figure is drawn first, for the final layout of the axes; the view is
    # orthographic, so a cut on the screen falls at the same share of a segment
    # in 3D
    fig.canvas.draw()

    def to_screen(xyz):
        # 3D points on the screen, in points
        xs, ys, _ = proj3d.proj_transform(*xyz.T, ax.get_proj())
        return ax.transData.transform(np.column_stack([xs, ys])) * 72 / fig.dpi

    coords = data.coords.numpy()
    circle = np.linalg.norm(coords[:, :2], axis=1).max() + MARGIN

    def onto_circle(xyz):
        # out from the z-axis onto the circle, in the plane of the net
        radial = circle / np.linalg.norm(xyz[:, :2], axis=1, keepdims=True)
        return np.column_stack([xyz[:, :2] * radial, xyz[:, 2]])

    centres = coords.reshape(6, 4, 3).mean(axis=1)
    # the nodes before a move, if given, count as ends too, although they are
    # not drawn: a translation's step slides by the pattern's own shift, so
    # after the move, the last one still ends where the outermost ring was
    nodes = coords if before is None else np.concatenate([coords, before.numpy()])
    ends = to_screen(np.concatenate([nodes, onto_circle(centres)]))
    # from a node's centre: its radius, with node_size=40 the area of its marker,
    # and the gap
    reach = np.sqrt(40) / 2 + GAP
    arrows = {tuple(arrow._end): arrow for arrow in ax.patches if isinstance(arrow, _Arrow3D)}
    for collection in ax.collections:
        if collection.get_zorder() != 5:
            continue
        steps = []
        for step in collection._segments3d:
            arrow = arrows[tuple(step[-1])]
            # a translation's step is a straight line between two points, a
            # rotation's an arc through several
            if len(step) == 2:
                lift = [0.0, LIFT, 0.0]
            else:
                step, lift = onto_circle(step), [0.0, 0.0, 0.0]
            # finely sampled, so that the cut falls close to the gap
            t = np.linspace(0, 1, 100, endpoint=False)[:, None]
            step = np.concatenate([a + t * (b - a) for a, b in zip(step[:-1], step[1:])] + [step[-1:]])
            screen = to_screen(step)
            clear = np.linalg.norm(screen[:, None] - ends, axis=2).min(axis=1) >= reach
            # from the first point clear of every end to the last one
            step = step[np.argmax(clear):len(step) - np.argmax(clear[::-1])] + lift
            steps.append(step)
            arrow._start, arrow._end = step[-3], step[-1]
        collection.set_segments(steps)

    path = f"docs/assets/{name}_dark.png" if DARK else f"docs/assets/{name}.png"
    plt.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)

    # bbox_inches="tight" doesn't crop 3D axes well, so crop the empty margin on
    # the saved image: found via the alpha channel on the transparent dark canvas,
    # and by diffing against white on the light one.
    img = Image.open(path).convert("RGBA")
    if DARK:
        bbox = img.getchannel("A").getbbox()
    else:
        bbox = ImageChops.difference(img.convert("RGB"), Image.new("RGB", img.size, "white")).getbbox()
    pad = 15
    left, upper, right, lower = bbox
    img = img.crop((max(left - pad, 0), max(upper - pad, 0), min(right + pad, img.width), min(lower + pad, img.height)))

    scale = 0.5
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    if DARK:
        img.save(path)
    else:
        img.convert("RGB").save(path)

    print(f"Saved {path}")


# one figure after each edge the tutorial adds: along the translation, around
# the rotation, and along both at once, with the edges added before lighter
data.add_edges_by_pattern_level(
    src_expansion_ids=[0], dest_expansion_ids=[0],
    src_level_positions=[[0, 0]], dest_level_positions=[[1, 0]],
)
save_figure("polar_net_pattern", num_old=0, seed_node=0, seed_edge=(0, 1))

num_old = int(data.directed_mask.sum())
data.add_edges_by_pattern_level(
    src_expansion_ids=[0], dest_expansion_ids=[0],
    src_level_positions=[[0, 0]], dest_level_positions=[[0, 1]],
)
save_figure("polar_net_pattern_rotational_edge", num_old, seed_edge=(0, 4))

num_old = int(data.directed_mask.sum())
data.add_edges_by_pattern_level(
    src_expansion_ids=[0], dest_expansion_ids=[0],
    src_level_positions=[[0, 0]], dest_level_positions=[[1, 1]],
)
save_figure("polar_net_pattern_diagonal_edge", num_old, seed_edge=(0, 5))

# and one after the tutorial moves the outermost ring outwards, with the moved
# nodes and the edges at them darker
before = data.coords.clone()
data.set_node_attr_by_pattern_level(
    "coords", expansion_ids=[0], level_positions=[[3, 0]],
    value=torch.tensor([[5.0, 0.0, 0.0]]), replicate_levels=[False, True],
)
save_figure("polar_net_pattern_moved_ring", before=before, seed_node=3)
