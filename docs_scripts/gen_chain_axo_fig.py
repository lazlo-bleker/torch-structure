import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageChops
from torch_structure.data.data import StructData

DARK = os.environ.get("TS_DARK") == "1"
BG = "none" if DARK else "white"

# data.plot() draws edges in a hardcoded red (plot_data() in
# torch_structure/plot/plot.py has no color-override parameter, and that
# function is shared well beyond these doc figures, so it's left untouched).
# Both figures here are 100% positive-force (all-tension) edges, so the
# whole structure renders in this one red -- recolored below, after
# data.plot() has already drawn it, to the site's orange brand accent.
RED_HEX = "#E40714"
ACCENT_HEX = "#F05F42" if DARK else "#E7461E"

def recolor_rgba(img, target_hex):
    # Alpha already IS the anti-aliasing blend fraction against the
    # transparent canvas -- just swap the constant foreground color and
    # leave alpha exactly as rendered.
    target = np.array(mcolors.to_rgb(target_hex)) * 255
    arr = np.array(img).astype(np.float64)
    arr[..., 0] = target[0]
    arr[..., 1] = target[1]
    arr[..., 2] = target[2]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA")

def recolor_rgb_on_white(img, orig_hex, target_hex):
    # No alpha channel here -- anti-aliasing is baked in as a linear
    # blend with the opaque white background. Recover the per-pixel blend
    # fraction against the known original color, then re-blend the target
    # color at that same fraction (using the G/B channels for the fraction
    # estimate since R has poor contrast between red and white).
    orig = np.array(mcolors.to_rgb(orig_hex)) * 255
    white = np.array([255.0, 255.0, 255.0])
    target = np.array(mcolors.to_rgb(target_hex)) * 255
    arr = np.array(img).astype(np.float64)
    a_est = (white - arr) / (white - orig)
    a = np.clip((a_est[..., 1] + a_est[..., 2]) / 2, 0, 1)
    new_rgb = a[..., None] * target + (1 - a[..., None]) * white
    return Image.fromarray(np.clip(new_rgb, 0, 255).astype(np.uint8), mode="RGB")

node_attrs = {"coords": torch.empty((0, 3), dtype=torch.float)}
edge_attrs = {"force": torch.empty((0, 1), dtype=torch.long)}
data = StructData(node_attrs=node_attrs, edge_attrs=edge_attrs)

num_nodes = 10

force = 10 * torch.ones(num_nodes - 1, 1, dtype=torch.long)

def f(x_unit_coord):
    height = 3 * x_unit_coord * (1 - x_unit_coord)
    return torch.hstack([x_unit_coord, torch.zeros_like(x_unit_coord), height])

node_attrs = {"coords": f}
edge_attrs = {"force": force}

data.add_chain(num_nodes, node_attrs=node_attrs, edge_attrs=edge_attrs)

fig = plt.figure(figsize=(10, 8), facecolor=BG)
ax = fig.add_subplot(111, projection="3d")
ax.view_init(elev=35.264, azim=45)  # isometric / axonometric view

data.plot(ax=ax, equal_axes=False, legend=False, lw_scale=0.5)

if DARK:
    # mplot3d draws opaque background "wall" panes behind the plot even
    # with set_axis_off() (default facecolor (0.95,0.95,0.95,0.5)) -- make
    # them fully transparent too, or they show up as a big solid rectangle
    # once composited onto the page's dark background.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax.patch.set_alpha(0.0)
    ax.set_facecolor((1, 1, 1, 0))

# size the bounding box to the actual per-axis data extent (the chain is
# planar in y, so a forced cubic box wastes a full extra x_range of empty
# depth) instead of the default equal_axes cube.
coords = f(x_unit_coord=torch.linspace(0, 1, num_nodes).unsqueeze(1))
pad = 0.05
x_min, y_min, z_min = coords.min(dim=0).values.tolist()
x_max, y_max, z_max = coords.max(dim=0).values.tolist()
x_range, y_range, z_range = x_max - x_min, max(y_max - y_min, 0.1), z_max - z_min
cx, cy, cz = (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2
ax.set_xlim3d(x_min - pad, x_max + pad)
ax.set_ylim3d(cy - y_range / 2 - pad, cy + y_range / 2 + pad)
ax.set_zlim3d(z_min - pad, z_max + pad)
ax.set_box_aspect((x_range + 2 * pad, y_range + 2 * pad, z_range + 2 * pad))
ax.set_proj_type("ortho")

path = "docs/assets/chain_axonometric_dark.png" if DARK else "docs/assets/chain_axonometric.png"
plt.savefig(path, dpi=150, facecolor=BG)

if DARK:
    # Same pipeline as the light-mode branch below, but working in RGBA
    # against a transparent canvas instead of RGB against a white one:
    # content is found via the alpha channel rather than by diffing a
    # solid background, and every canvas we pad onto starts fully
    # transparent instead of solid white.
    img = Image.open(path).convert("RGBA")

    bbox = img.split()[-1].getbbox()
    pad = 15
    if bbox:
        left, upper, right, lower = bbox
        left = max(left - pad, 0)
        upper = max(upper - pad, 0)
        right = min(right + pad, img.width)
        lower = min(lower + pad, img.height)
        img = img.crop((left, upper, right, lower))

    # the chain's silhouette isn't symmetric within its own bounding box (the
    # isometric skew makes one end reach further than the other), so pad
    # asymmetrically to put the drawn pixels' centroid at the horizontal center.
    arr = np.array(img)
    content_mask = arr[:, :, 3] > 0
    ys, xs = np.where(content_mask)
    cx = xs.mean()
    W, H = img.size
    pad_left = max(W - 2 * cx, 0)
    pad_right = max(2 * cx - W, 0)
    new_img = Image.new("RGBA", (int(W + pad_left + pad_right), H), (0, 0, 0, 0))
    new_img.paste(img, (int(pad_left), 0))

    # shrink slightly (extra margin) and nudge right (more margin on the left
    # than the right); vertical margin kept much smaller than horizontal
    W2, H2 = new_img.size
    margin_x = int(0.24 * max(W2, H2))
    margin_top = int(0.1 * max(W2, H2))
    margin_bottom = int(0.06 * max(W2, H2))
    shift = int(0.5 * W2)
    left_margin = margin_x + shift
    right_margin = max(margin_x - shift, int(0.02 * W2))
    final = Image.new("RGBA", (W2 + left_margin + right_margin, H2 + margin_top + margin_bottom), (0, 0, 0, 0))
    final.paste(new_img, (left_margin, margin_top))

    scale = 0.5
    final = final.resize((int(final.width * scale), int(final.height * scale)), Image.LANCZOS)
    final = recolor_rgba(final, ACCENT_HEX)
    final.save(path)
else:
    # matplotlib's bbox_inches="tight" doesn't crop 3D axes well (the viewport
    # always reserves a full square regardless of how much of it the data fills),
    # so auto-crop the surrounding whitespace directly on the saved image.
    img = Image.open(path).convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    pad = 15
    if bbox:
        left, upper, right, lower = bbox
        left = max(left - pad, 0)
        upper = max(upper - pad, 0)
        right = min(right + pad, img.width)
        lower = min(lower + pad, img.height)
        img = img.crop((left, upper, right, lower))

    # the arc's silhouette isn't symmetric within its own bounding box (the
    # isometric skew makes one end reach further than the other), so pad
    # asymmetrically to put the red pixels' centroid at the horizontal center.
    # (Only balanced horizontally -- balancing vertically too would add a lot of
    # top whitespace, since the arc's two segments overlap more in the upper
    # half of its bounding box.)
    arr = np.array(img)
    red_mask = (arr[:, :, 0] > 180) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100)
    ys, xs = np.where(red_mask)
    cx = xs.mean()
    W, H = img.size
    pad_left = max(W - 2 * cx, 0)
    pad_right = max(2 * cx - W, 0)
    new_img = Image.new("RGB", (int(W + pad_left + pad_right), H), (255, 255, 255))
    new_img.paste(img, (int(pad_left), 0))

    # shrink slightly (extra margin) and nudge right (more margin on the left
    # than the right); vertical margin kept much smaller than horizontal
    W2, H2 = new_img.size
    margin_x = int(0.24 * max(W2, H2))
    margin_top = int(0.1 * max(W2, H2))
    margin_bottom = int(0.06 * max(W2, H2))
    shift = int(0.5 * W2)
    left_margin = margin_x + shift
    right_margin = max(margin_x - shift, int(0.02 * W2))
    final = Image.new("RGB", (W2 + left_margin + right_margin, H2 + margin_top + margin_bottom), (255, 255, 255))
    final.paste(new_img, (left_margin, margin_top))

    scale = 0.5
    final = final.resize((int(final.width * scale), int(final.height * scale)), Image.LANCZOS)
    final = recolor_rgb_on_white(final, RED_HEX, ACCENT_HEX)
    final.save(path)

print(f"Saved {path}")
