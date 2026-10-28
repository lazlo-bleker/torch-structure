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
# how far the pattern's arrows are lifted off the edges they run along, in y
LIFT = 0.1

def faded(hex_color, amount=0.45 if DARK else 0.55):
    # blend toward the page's background: lighter on the light page, dimmer
    # on the dark one, a little less there, where it would otherwise all but
    # vanish
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

translation = data.create_translational_pattern(4)
data.add_patterns({"translation": translation})

data.add_nodes_by_pattern(pattern="translation", coords=torch.tensor([[0.0, 0.0, 0.0]]))
data.add_edges_by_pattern(edge_indices=torch.tensor([[0], [1]]))
before = data.coords.clone().numpy()

mask = torch.tensor([True, False, False, False])
data.set_node_attr_by_pattern("coords", mask=mask, value=torch.tensor([[0.0, 1.0, 0.0]]))
after = data.coords.numpy()


def save_figure(name, moved=False, seed_node=None, seed_edge=None):
    # seed_node, seed_edge: the node, and the edge as its two nodes, given to
    # the step the figure shows
    fig = plt.figure(figsize=(6, 6), facecolor=BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=90, azim=-90)  # plan / top view, x to the right and y up

    # the nodes and their pattern before the move, and if moved, these in
    # faded colors and the ones after the move in solid colors, with
    # plot_pattern() drawing on the same axes twice
    data.plot_pattern(
        ax=ax, coords=before, color=faded(ACCENT_HEX) if moved else ACCENT_HEX,
        node_size=40, lw_scale=0.7, legend=False, show_labels=False,
    )
    num_before = len(ax.collections) if moved else 0
    if moved:
        data.plot_pattern(ax=ax, color=ACCENT_HEX, node_size=40, lw_scale=0.7, legend=False, show_labels=False)

    # plot_pattern() fits the limits to what it drew last, i.e. only one of the
    # rows, so fit them to both rows, before and after the move, in every figure,
    # with one scale on all axes. mplot3d sizes the box by its diagonal, so the
    # long, flat box needs zooming out to fit the axes from the top, which also
    # gives the same scale as the rotational figures
    points = np.concatenate([before, after])
    low, high = points.min(axis=0), points.max(axis=0)
    ax.set_xlim3d(low[0], high[0])
    ax.set_ylim3d(low[1], high[1])
    ax.set_zlim3d(low[2] - 0.1, high[2] + 0.1)
    ax.set_box_aspect(tuple(np.maximum(high - low, 0.2)), zoom=0.55)

    # plot_pattern() draws the edges at z-order 2 and the nodes at 3, both in
    # hardcoded greys, and the steps of the pattern at 5. After the move, the
    # seed edge and node, if given, are drawn darker and larger
    edges = data.edge_index[:, data.directed_mask.view(-1)]
    for i, collection in enumerate(ax.collections):
        if collection.get_zorder() == 2 and i >= num_before and seed_edge is not None:
            mark_seed(collection, is_edge(edges, *seed_edge).tolist(), [NODE_HEX] * edges.shape[1])
        elif collection.get_zorder() == 3 and i >= num_before and seed_node is not None:
            mark_seed(collection, (torch.arange(data.num_nodes) == seed_node).tolist(), [NODE_HEX] * data.num_nodes)
        elif collection.get_zorder() in (2, 3):
            collection.set_color(faded(NODE_HEX) if i < num_before else NODE_HEX)

    if DARK:
        # mplot3d's background panes and the axes' own background patch are
        # opaque, make them transparent for the page's dark background
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor((1, 1, 1, 0))
        ax.patch.set_alpha(0.0)
        ax.set_facecolor((1, 1, 1, 0))

    # plot_pattern() trims each step by a share of its length, which leaves short
    # steps touching their nodes, so cut every step back to GAP points clear of
    # the nodes it runs between, on the screen, lift it off the edge it runs
    # along, and move its arrowhead along. The figure is drawn first, for the
    # final layout of the axes; the view is orthographic, so a cut on the screen
    # falls at the same share of a segment in 3D
    fig.canvas.draw()

    def to_screen(xyz):
        # 3D points on the screen, in points
        xs, ys, _ = proj3d.proj_transform(*xyz.T, ax.get_proj())
        return ax.transData.transform(np.column_stack([xs, ys])) * 72 / fig.dpi

    ends = to_screen(np.concatenate([before, after]) if moved else before)
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
            # finely sampled, so that the cut falls close to the gap
            t = np.linspace(0, 1, 100, endpoint=False)[:, None]
            step = np.concatenate([a + t * (b - a) for a, b in zip(step[:-1], step[1:])] + [step[-1:]])
            screen = to_screen(step)
            start, end = (ends[np.argmin(np.linalg.norm(ends - point, axis=1))] for point in (screen[0], screen[-1]))
            clear_of_start = np.linalg.norm(screen - start, axis=1) >= reach
            clear_of_end = np.linalg.norm(screen - end, axis=1) >= reach
            # from the first point clear of the start to the last one clear of the end
            step = step[np.argmax(clear_of_start):len(step) - np.argmax(clear_of_end[::-1])] + [0.0, LIFT, 0.0]
            steps.append(step)
            arrow._start, arrow._end = step[-3], step[-1]
        collection.set_segments(steps)

    if moved:
        # an arrow for the moved node, drawn in 2D between the projected nodes: a
        # 3D quiver's arrowhead lies in the vertical plane, so it vanishes from
        # the top
        xs, ys, _ = proj3d.proj_transform(*np.stack([before[0], after[0]], axis=1), ax.get_proj())
        ax.annotate(
            "", xy=(xs[1], ys[1]), xytext=(xs[0], ys[0]), zorder=4,
            arrowprops=dict(arrowstyle="->", color=NODE_HEX, lw=1.2, shrinkA=10, shrinkB=10),
        )

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


# one figure for each step of the tutorial: the nodes and edges before the
# move, and the move, both at one scale
save_figure("translational_pattern_edges", seed_node=0, seed_edge=(0, 1))
save_figure("translational_pattern", moved=True, seed_node=0)
